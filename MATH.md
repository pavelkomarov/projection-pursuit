This markdown is intended to give readers a full explanation of how multivariate Projection Pursuit Regression (PPR) and univariate Classification work. It is as much for me as for you, because every time I have to dig in to the bones of this thing to work on it, I end up having to refer to the original paper's somewhat ambiguous, derivationless equations, which can end up being a gigantic waste of time. This document also serves as a way to decrease the load of comments in the code, which currently take up maybe 2/3 of skpp.py.

The math here is written in LaTeX, which I understand github doesn't render natively. You may need a plugin or to compile this document with pdflatex to view it comfortably.

PPR is a statistical model of the form:

		$$y_i = sum j=1 to r (f_j(x_i*alpha_j) * beta_j^T)$$

	where:
		*$i$ iterates examples, the rows of input and output matrices
		*$j$ iterates the number of terms in the PPR "additive model"
		*$r$ is the total number of projections and functions in the PPR
		*$y_i$ is a $d$-dimensional vector, the $i$th row in an output matrix $Y \in \mathbb{R}^{d \times n}$
		*$x_i$ is a $p$-dimensional vector, the $i$th row of an input matrix $X \in \mathbb{R}^{p \times n}$
		*$alpha_j$ is the $j$th projection vector in the mdoel, a $p$-dimensional vector inner-producted with $x_i$
		*$f_j$ is the $j$th function in the model, mapping from $\mathbb{R}^1 \rightarrow \mathbb{R}^1$
		*$beta_j^T$ is the transpose of $beta_j$, a $d$-dimensional vector outer	producted with the result of $f_j$ to yield a result in the output space
		* $*$ is a product, the inner product for $x_i*alpha_j$, the outer product for $f_j * beta_j^T$
