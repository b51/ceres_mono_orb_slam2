### Some neccessary equations before Jacobian calculation
**1. With the property of Lie Algebra Adjoint, [Reference](https://blog.csdn.net/heyijia0327/article/details/51773578)**
$$\mathbf{S * exp(\hat{\zeta}) * S^{-1} = exp[(Adj(S) * \zeta)^{\Lambda}]}$$
We can get equations below
$$\mathbf{S * exp(\hat{\zeta}) = exp[(Adj(S) * \zeta)^{\Lambda}] * S}$$
and
$$\mathbf{exp(\hat{\zeta}) * S^{-1} = S^{-1} * exp[(Adj(S) * \zeta)^{\Lambda}]}$$

**2. Baker-Campbell-Hausdorf equations, [STATE ESTIMATION FOR ROBOTICS](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) P.234**

$\hspace{2cm}\mathbf{ln(S_1 S_2)^{v} = ln(exp[\hat{\xi_1]}exp[\hat{\xi_2]})^{v}}$

$\hspace{4cm}\mathbf{ = \xi_1 + \xi_2 + \frac{1}{2} \xi_1^{\lambda} \xi_2 + \frac{1}{12}\xi_1^{\lambda} \xi_1^{\lambda} \xi_2 + \frac{1}{12}\xi_2^{\lambda} \xi_2^{\lambda} \xi_1 + \dots}$

$\hspace{4cm}\mathbf{\approx J_l(\xi_2)^{-1}\xi_1 + \xi_2}\hspace{2cm}$ (if $\xi_1 small$)

$\hspace{4cm}\mathbf{\approx \xi_2 + J_r(\xi_1)^{-1}\xi_2}\hspace{2cm}$ (if $\xi_2 small$)

With
$\hspace{3cm}\mathbf{J_l(\xi)^{-1} = \sum_{n = 0}^{\infty} \frac{B_n}{n!} (\xi^{\lambda})^{n}}$

$\hspace{3cm}\mathbf{J_r(\xi)^{-1} = \sum_{n = 0}^{\infty} \frac{B_n}{n!} (-\xi^{\lambda})^{n}}$

With$\hspace{1cm}B_0 = 1, B_1 = -\frac{1}{2}, B_2 = \frac{1}{6}, B_3 = 0, B_4 = -\frac{1}{30}\dots$, $\hspace{5mm}\mathbf{\xi^{\lambda} = adj(\xi)}$, $\hspace{5mm}$ is adjoint matrix of $\xi$

**3. Adjoint Matrix of sim(3)**

a) Main property of adjoint matrix on Lie Algebras, [Reference: LIE GROUPS AND LIE ALGEBRAS, 1.6](http://www.math.jhu.edu/~fspinu/423/7.pdf)
$$\mathbf{[x, y] = adj(x)y}$$

b) sim3 Lie Brackets, [Reference: Local Accuracy and Global Consistency for Efficient Visual SLAM, P.184, A.3.4](https://www.doc.ic.ac.uk/~ajd/Publications/Strasdat-H-2012-PhD-Thesis.pdf):
$$\mathbf{[x, y] = [\begin{bmatrix} \nu \newline \omega \newline \sigma\end{bmatrix} \begin{bmatrix} \tau \newline \varphi \newline \varsigma \end{bmatrix}] = \begin{bmatrix}\omega \times \tau + \nu \times \varphi + \sigma\tau - \varsigma\nu \newline \omega \times \varphi \newline  0 \end{bmatrix}}$$
$$\mathbf{= \begin{bmatrix}(\hat{\omega} + \sigma I)\tau + \nu \times \varphi - \varsigma\nu \newline \omega \times \varphi \newline  0 \end{bmatrix}}$$
$$\hspace{-4cm} = \mathbf{adj(x) y}$$

We can get $\hspace{4cm}\mathbf{\xi^{\lambda} = adj(\xi) = \begin{bmatrix} (\hat{\omega} + \sigma I) & \hat{\nu} & -{\nu} \newline 0 & \hat{\omega} & 0 \newline 0 & 0 & 0 \end{bmatrix}}$

### Left multiplication/Right multiplication for pose update
sim(3) update with **Left multiplication/Right multiplication** has affect on Jacobian calculation, Formula derivation below used Right multiplication as example

### Jacobian Calculation of sim(3)
$$\mathbf{error = S_{ji}  S_{iw}  S_{jw}^{-1}}$$

Derivation of $\mathbf{Jacobian_i}$

$\hspace{6cm}\mathbf{ln(error(\xi_i + \delta_i))^v = ln(S_{ji}S_{iw}exp(\hat{\delta_i})S_{jw}^{-1})^{v}}$

$\hspace{10cm}\mathbf{= ln(S_{ji}S_{iw}S_{jw}^{-1}exp[(Adj(S_{jw}){\delta_i})^{\Lambda}])^{v}}$

$\hspace{10cm}\mathbf{= ln(exp(\xi_{error}) \cdot exp[(Adj(S_{jw}){\delta_i})^{\Lambda}])^{v}}$

$\hspace{10cm}\mathbf{= \xi_{error} + J_r(\xi_{error})^{-1} Adj(S_{jw}){\delta_i}}$

$$\mathbf{Jacobian_i = \frac{\partial ln(error)^v}{\partial \delta_i} = \lim_{\delta_i \to 0} \frac{ln(error(\xi_i + \delta_i))^v - ln(error)^v}{\delta_i} = J_r(\xi_{error})^{-1} Adj(S_{jw}) }$$

Same to $\mathbf{Jacobian_j}$

$\hspace{6cm}\mathbf{ln(error(\xi_i + \delta_j)) = ln(S_{ji}S_{iw}(S_{jw}exp(\hat{\delta_j}))^{-1})^{v}}$
$\hspace{10cm}\mathbf{= ln(S_{ji}S_{iw}exp(-\hat{\delta_j})S_{jw}^{-1})^{v}}$

$\hspace{10cm}\mathbf{= ln(exp(\xi_{error}) \cdot exp[(Adj(S_{jw})(-\delta_j))^{\Lambda}])^{v}}$

$\hspace{10cm}\mathbf{= \xi_{error} - J_r(\xi_{error})^{-1} Adj(S_{jw}){\delta_j}}$

$$\mathbf{Jacobian_j = \frac{\partial ln(error)^v}{\partial \delta_j} = \lim_{\delta_j \to 0} \frac{ln(error(\xi_i + \delta_j))^v - ln(error)^v}{\delta_j} = -J_r(\xi_{error})^{-1} Adj(S_{jw}) }$$

