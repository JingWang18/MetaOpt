import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from math import pi
from scipy.special import logsumexp


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    def __init__(self, n_components, n_features, covariance_type="full",
                 eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None, share_covariance=True):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components # k
        self.n_features = n_features # d

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.share_covariance = share_covariance
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self._init_params()


    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
                    requires_grad=False
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)
        self.params_fitted = False

        #if self.share_covariance:
        #    self.var.data = torch.sum(
        #        self.var.data * self.pi.data, dim=1, keepdim=True).repeat(1, self.n_components, 1)


    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x


    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic


    def fit(self, x, delta=1e-3, n_iter=50, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(self.n_components,
                    self.n_features,
                    covariance_type=self.covariance_type,
                    mu_init=self.mu_init,
                    var_init=self.var_init,
                    eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

            #if i % 10 == 0:
            #    print(f'{i}/{n_iter}')

        self.params_fitted = True


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            unnorm_p_k = torch.exp(weighted_log_prob)
            p_k = torch.squeeze(unnorm_p_k / (unnorm_p_k.sum(1, keepdim=True)))
            return p_k
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))


    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def get_nearest_samples(self, x):
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        nearest_samples = torch.argmax(weighted_log_prob, dim=0).squeeze(-1)

        # check overlap
        #np_nearest_samples = nearest_samples.detach().cpu().numpy()

        topk = torch.topk(weighted_log_prob, k=weighted_log_prob.shape[1], dim=0).indices.squeeze(-1)

        nearest_samples = []
        for i in range(weighted_log_prob.shape[1]):
            for idx in topk[:,i]:
                if idx not in nearest_samples:
                    nearest_samples.append(idx)
                    break

        return torch.tensor(nearest_samples)

    def get_closest_k_samples(self, x, p=1.0):
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        assignments = torch.squeeze(
            torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

        closest_k_samples = []
        indices = torch.arange(len(assignments)).to(x.device)
        for cls in torch.unique(assignments):
            cls_indices = indices[assignments == cls]
            k = int(len(cls_indices) * p)
            if len(cls_indices) < 5:
                print(f'num points for assignment {cls}: {len(cls_indices)}')

            if k <= 1:
                idx = [-1]
            else:
                idx = torch.topk(
                    weighted_log_prob[assignments == cls, cls], k=k, dim=0, largest=False).indices[-1]
            closest_k_samples.append(cls_indices[idx])

        try:
            torch.cat(closest_k_samples)
        except:
            import IPython; IPython.embed()
        return torch.cat(closest_k_samples)


    def get_farthest_samples(self, x):
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        assignments = torch.squeeze(
            torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

        farthest_samples = []
        indices = torch.arange(len(assignments)).to(x.device)
        for cls in torch.unique(assignments):
            cls_indices = indices[assignments == cls]
            idx = torch.argmin(
                weighted_log_prob[assignments == cls, cls], dim=0)
            farthest_samples.append(cls_indices[idx])

        return torch.cat(farthest_samples)


    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in np.arange(self.n_components)[counts > 0]:
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y


    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score


    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            precision = torch.inverse(var).float()
            d = x.shape[-1]

            log_2pi = d * np.log(2. * pi)

            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det



    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0,k]))).sum()

        return log_det.unsqueeze(-1)


    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp


    def _m_step(self, x, log_resp, batch_size=128):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = torch.zeros(1, log_resp.shape[1], x.shape[-1], x.shape[-1]).to(x.device)
            for i in range(0, x.shape[0], batch_size):
                var += torch.sum(
                    (x[i:i+batch_size] - mu).unsqueeze(-1).matmul((x[i:i+batch_size] - mu).unsqueeze(-2)) * resp[i:i+batch_size].unsqueeze(-1), dim=0, keepdim=True)
            var = (var / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps).double()
            #var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
            #                keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps
        elif self.covariance_type == "diag":
            if self.share_covariance:
                var = (resp * (x - mu) ** 2).sum((0, 1), keepdim=True) / torch.sum(resp)
                var = var.repeat(1, self.n_components, 1)
            else:
                x2 = (resp * x * x).sum(0, keepdim=True) / pi
                mu2 = mu * mu
                xmu = (resp * mu * x).sum(0, keepdim=True) / pi
                var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var


    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)


    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)


    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var


    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi


    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]

            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0)*(x_max - x_min) + x_min)


class MixReg:
    def __init__(self, K, max_iter=100, tol=1e-4):
        """
        Initializes a mixture of regression model with K subpopulations.
        Args:
            K (int): Number of subpopulations.
            max_iter (int): Maximum number of EM iterations (default: 100).
            tol (float): Convergence tolerance for the log-likelihood (default: 1e-4).
        """
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.beta = None
        self.sigma2 = None
        self.loglik = None

    def fit(self, X, y):
        """
        Fits the mixture of regression model using the EM algorithm.
        Args:
            X (torch.Tensor): Design matrix of shape (n, p).
            y (torch.Tensor): Response vector of shape (n,).

        Returns:
            dict: Dictionary containing the estimated model parameters.
        """
        n, p = X.shape
        alpha = torch.full((self.K,), 1 / self.K, requires_grad=False)
        beta = torch.randn((p, self.K), requires_grad=False)
        sigma2 = torch.full((self.K,), y.var().item(), requires_grad=False)

        log_likelihood = -float('inf')
        for i in range(1, self.max_iter + 1):
            # E-step: Compute the posterior probabilities
            posterior_probs = torch.zeros((n, self.K))
            for k in range(self.K):
                mu = X @ beta[:, k]
                dist = Normal(mu, torch.sqrt(sigma2[k]))
                posterior_probs[:, k] = dist.log_prob(y.squeeze()) + torch.log(alpha[k])
            log_likelihood_new = torch.logsumexp(posterior_probs, dim=1).sum().item()
            posterior_probs = torch.exp(posterior_probs - log_likelihood_new)

            # M-step: Update the model parameters
            alpha_new = posterior_probs.mean(dim=0)
            for k in range(self.K):
                weights = posterior_probs[:, k]
                X_weighted = X * weights.view(-1, 1)
                beta[:, k] = torch.pinverse(X_weighted) @ (y * weights)
                residual = y - X @ beta[:, k]
                sigma2[k] = (weights * residual.pow(2)).sum() / weights.sum()

            # Check for convergence
            if abs(log_likelihood_new - log_likelihood) < self.tol:
                break
            log_likelihood = log_likelihood_new

        self.alpha = alpha_new
        self.beta = beta
        self.sigma2 = sigma2
        self.loglik = log_likelihood

        # Pack the model parameters into a dictionary
        model_params = {'alpha': alpha_new, 'beta': beta, 'sigma2': sigma2}

        return model_params

    def predict(self, X):
        return

    def mixreg_cov(self, X, model_params):
        """
        Computes the covariance matrix of the estimated regression coefficients.
        Args:
            X (torch.Tensor): Design matrix of shape (n, p).
            model_params (dict): Dictionary containing the trained model parameters.
        Returns:
            torch.Tensor: Covariance matrix of shape (p, p).
        """
        n, p = X.shape
        K = len(model_params['alpha'])

        # Compute the covariance matrix
        V = torch.zeros((p, p))
        for k in range(K):
            weights = model_params['alpha'][k] * Normal(X @ model_params['beta'][:, k],
                                                        torch.sqrt(model_params['sigma2'][k])).log_prob(X @ model_params['beta'][:, k]).exp()
            X_weighted = X * weights.view(-1, 1)
            V += torch.pinverse(X_weighted.t() @ X_weighted) / model_params['sigma2'][k]

        V /= n
        return V



class MixRegEM:
    def __init__(self, n_components, n_features, device='gpu', reg_coef=1e-6, n_iter=100, tol=1e-4):
        self.K = n_components
        self.n_features = n_features
        self.reg_coef = reg_coef
        self.n_iter = n_iter
        self.tol = tol
        self.device = device

        self.means = torch.randn(n_components, n_features).to(device)
        self.covs = [torch.eye(n_features).to(device) for _ in range(n_components)]
        self.weights = (torch.ones(n_components) / n_components).to(device)
        self.responsibilities = None

    def fit(self, X, y):
        n, p = X.shape
        alpha = torch.full((self.K,), 1 / self.K, requires_grad=False).to(self.device)
        beta = torch.randn((p, self.K, y.shape[1]), requires_grad=False).to(self.device)
        sigma2 = torch.full((self.K, y.shape[1]), 1 / self.K, requires_grad=False).to(self.device)

        log_likelihood = -float('inf')
        for i in range(1, self.n_iter + 1):
            # E-step: Compute the posterior probabilities
            posterior_probs = torch.zeros((n, self.K)).to(self.device)
            for k in range(self.K):
                mu = X @ beta[:, k]
                dist = MultivariateNormal(mu, torch.diag(torch.sqrt(sigma2[k])))
                posterior_probs[:, k] = dist.log_prob(y.squeeze()) + torch.log(alpha[k])
            log_likelihood_new = torch.logsumexp(posterior_probs, dim=1, keepdim=True)
            posterior_probs = torch.exp(posterior_probs - log_likelihood_new)

            # M-step: Update the model parameters
            alpha_new = posterior_probs.mean(dim=0)
            for k in range(self.K):
                weights = posterior_probs[:, k]
                X_weighted = X * weights.view(-1, 1)
                try:
                    beta[:, k] = torch.pinverse(X_weighted) @ (y * weights)
                except:
                    import IPython; IPython.embed(); exit()

                residual = y - X @ beta[:, k]
                sigma2[k] = (weights * residual.pow(2)).sum(dim=0) / weights.sum()


        #n_samples = X.shape[0]
        #y = torch.tensor(y, dtype=torch.float32)

        ## Run EM algorithm
        #for i in range(self.n_iter):
        #    # E-step: compute responsibilities
        #    log_probs = []
        #    for k in range(self.n_components):
        #        mean = self.means[k]
        #        cov = self.covs[k]
        #        log_prob = torch.distributions.MultivariateNormal(mean, cov).log_prob(X)
        #        log_probs.append(log_prob)
        #    log_probs = torch.stack(log_probs, dim=1)
        #    log_weighted_probs = log_probs + torch.log(self.weights)
        #    max_log_probs, _ = torch.max(log_weighted_probs, dim=1, keepdim=True)
        #    exp_probs = torch.exp(log_weighted_probs - max_log_probs)
        #    self.responsibilities = exp_probs / torch.sum(exp_probs, dim=1, keepdim=True)

        #    # M-step: update parameters
        #    for k in range(self.n_components):
        #        resp = self.responsibilities[:, k]
        #        total_resp = torch.sum(resp)
        #        self.weights[k] = total_resp / n_samples
        #        self.means[k] = torch.sum(resp.view(-1, 1) * X, dim=0) / total_resp
        #        diff = X - self.means[k]
        #        self.covs[k] = (diff * resp.view(-1, 1)).t() @ diff / total_resp
        #        self.covs[k] += torch.eye(self.n_features).to(self.device) * self.reg_coef
        #        if torch.any(self.covs[k] < 0):
        #            import IPython; IPython.embed()
        #    self.weights /= torch.sum(self.weights)

        #    # Check for convergence
        #    if i > 0:
        #        max_diff = torch.max(torch.abs(prev_means - self.means))
        #        if max_diff < self.tol:
        #            break
        #    prev_means = self.means.clone()

    def predict(self, X):
        log_probs = []
        for k in range(self.n_components):
            mean = self.means[k]
            cov = self.covs[k]
            log_prob = torch.distributions.MultivariateNormal(mean, cov).log_prob(X)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs, dim=1)
        log_weighted_probs = log_probs + torch.log(self.weights)
        return torch.exp(log_weighted_probs)
        #max_log_probs, argmax = torch.max(log_weighted_probs, dim=1)
        #return argmax.numpy(), max_log_probs.numpy()

    def mixreg_cov(self):
        covs = []
        for k in range(self.n_components):
            cov = torch.inverse(self.covs[k])
            covs.append(cov)
        return torch.stack(covs)

