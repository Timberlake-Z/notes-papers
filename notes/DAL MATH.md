# DAL MATH

## Recall the proof of DAL algorithm

1. ood risk model
   $$
   R_{\mathcal{D}}(\mathbf{w}) = R_{\mathcal{I}}(\mathbf{w}) + \alpha R_{\mathcal{O}}(\mathbf{w}),
   $$

2. sample based risk estimation with data we have (id) 
   $$
   R_{\mathcal{I}}(\mathbf{w}) = \mathbb{E}_{(\mathbf{x}, y) \sim D_{X_{\mathcal{I}}Y_{\mathcal{I}}}} \ell(f_{\mathbf{w}}; \mathbf{x}, y)
   \quad \text{and} \quad
   \hat{R}_{\mathcal{I}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_{\mathbf{w}}; \mathbf{x}^{i}_{\mathcal{I}}, y^{i}_{\mathcal{I}})\\
   R_{\mathcal{A}}(\mathbf{w}) = \mathbb{E}_{\mathbf{x} \sim D_{X_{\mathcal{A}}}} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x})
   \quad \text{and} \quad
   \hat{R}_{\mathcal{A}}(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x}^{i}_{\mathcal{A}})
   $$

3. Learning goal
   $$
   \min_{\mathbf{w} \in \mathcal{W}} \left[ \hat{R}_{\mathcal{I}}(\mathbf{w}) + \alpha \hat{R}_{\mathcal{A}}(\mathbf{w}) \right].
   $$

4. Consider Distributional robust optimization (DRO), extend the scope of estimated ood data to distributions within the range with defined distance measurement

5. New Learning goal (DRO imported)
   $$
   \min_{\mathbf{w} \in \mathcal{W}} \left[ \hat{R}_{\mathcal{I}}(\mathbf{w}) + \alpha \sup_{D_{X'} \in \mathcal{D}} \mathbb{E}_{\mathbf{x} \sim D_{X'}} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x}) \right], \quad \text{subject to } \hat{D}_{X_{\mathcal{A}}} \in \mathcal{D},
   $$

6. define the origin as the sphere centered at the auxiliary ood dataset. (Wasserstein ball ), Distance measurement (distribution ~ distribution )
   $$
   \hat{R}_{\mathcal{O}}(\mathbf{w}; \rho) = \sup_{W_c(D_{X'}, \hat{D}_{X_{\mathcal{A}}}) \leq \rho} \mathbb{E}_{\mathbf{x} \sim D_{X'}} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x}).\\
   W_c(\mu, \nu) = \inf_{\gamma \in \Pi(\mu, \nu)} \mathbb{E}_{(x, y) \sim \gamma} \left[ c(x, y) \right]
   $$

7. New goal
   $$
   \min_{\mathbf{w} \in \mathcal{W}} \left[ 
   \hat{R}_{\mathcal{I}}(\mathbf{w}) + \alpha \cdot \sup_{D_{X'} \in \mathcal{D}} 
   \mathbb{E}_{\mathbf{x} \sim D_{X'}} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x}) 
   \right], \quad \text{where } \mathcal{D} = \left\{ D_{X'} : W_c(D_{X'}, \hat{D}_{X_{\mathcal{A}}}) \leq \rho \right\}
   $$
   

8. Lagrange Dual, (Lagrange multiplier)
   $$
   \mathcal{L}(D_{X'}, \lambda) = 
   \mathbb{E}_{\mathbf{x} \sim D_{X'}} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x}) 
   + \lambda \left( \rho - W_c(D_{X'}, \hat{D}_{X_{\mathcal{A}}}) \right)
   $$
   

9. Saddle point problem
   $$
   \sup_{D_{X'}} \inf_{\lambda \geq 0} \left[
   \mathbb{E}_{\mathbf{x} \sim D_{X'}} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x}) 
   + \lambda \left( \rho - W_c(D_{X'}, \hat{D}_{X_{\mathcal{A}}}) \right)
   \right]
   $$

10. Sion’s Minimax Theorem
    $$
    \sup_{y \in Y} \inf_{x \in X} \mathcal{L}(x, y)
    =
    \inf_{x \in X} \sup_{y \in Y} \mathcal{L}(x, y)
    $$

11. Converted Problem
    $$
    \inf_{\lambda \geq 0} \left[
    \lambda \rho + \sup_{D_{X'}} 
    \left( \mathbb{E}_{\mathbf{x} \sim D_{X'}} \ell_{\mathrm{OE}}(f_{\mathbf{w}}; \mathbf{x}) 
    - \lambda W_c(D_{X'}, \hat{D}_{X_{\mathcal{A}}}) - \alpha\frac{d(D_{X'},ID)}{d(D_{A},ID)} \right)
    \right]\\
    
    \text{recall the def of Wasserstein}\\
    
    W_c(D_{X'}, \hat{D}) = \inf_{\pi \in \Pi(D_{X'}, \hat{D})} \mathbb{E}_{(x, z) \sim \pi} c(x, z)
    $$

12. Unfold the def of Expectation of loss function
    $$
    \mathbb{E}_{\mathbf{x} \sim D_{X'}} \ell(f_{\mathbf{w}}; \mathbf{x}) 
    = \mathbb{E}_{(x, z) \sim \pi} \ell(f_{\mathbf{w}}; z)
    $$

13. merge Expectation and Wasserstein distance into one coupling, Blanchet & Murthy dual
    $$
    \inf_{\lambda \geq 0} \left[
    \lambda \rho + \sup_{\pi \in \Pi(\hat{D}_{X_{\mathcal{A}}}, \cdot)} 
    \int \left( \ell_{\mathrm{OE}}(f_{\mathbf{w}}; z) - \lambda c(x, z) \right) \, d\pi(x, z)
    \right]
    
    \\
    \text{which is}
    \\
    \inf_{\lambda \geq 0} \left[
    \lambda \rho + \sup_{\pi \in \Pi(\hat{D}, \cdot)} 
    \mathbb{E}_{(x, z) \sim \pi} \left[ \ell_{\mathrm{OE}}(f_{\mathbf{w}}; z) - \lambda c(x, z) \right]
    \right]
    $$
    

14. Pointwise version, sampled data distribution +  **strictly convex** 
    $$
    \inf_{\lambda \geq 0} \left[
    \lambda \rho + \frac{1}{m} \sum_{i=1}^{m} 
    \sup_{z \in \mathcal{X}} \left( 
    \ell_{\mathrm{OE}}(f_{\mathbf{w}}; z) - \lambda c(x_i, z)
    \right)
    \right]
    $$
    

15. Summary
    $$
    \min_{\mathbf{w}} \left\{
    \hat{R}_{\mathcal{I}}(\mathbf{w}) 
    + \alpha \cdot \inf_{\lambda \geq 0} \left[
    \lambda \rho + \frac{1}{m} \sum_{i=1}^{m} 
    \sup_{z \in \mathcal{X}} \left( 
    \ell_{\mathrm{OE}}(f_{\mathbf{w}}; z) - \lambda c(x_i, z)
    \right)
    \right]
    \right\}
    $$

    ****

    **Input**: ID and OOD samples from $D_{X_1 Y_1}$ and $D_{X_{\mathcal{A}}}$

    **for** \( st = 1 \) to `num_step` **do**  
     Sample $ S_B , T_B $ from  $D_{X_1 Y_1}$ ,  $D_{X_{\mathcal{A}}}$ 
     Initialize:  
     $$ p^i \sim \mathcal{N}(0, \sigma I),\quad \forall i \in \{1, \dots, |T_B|\} $$

     **for** \( `se` = 1 \) to `num_search` **do**  
      Update direction:  
      $$ \psi^i \leftarrow \nabla_{p^i} \left[ \ell_{\mathrm{OE}}(h(e(x^i_{\mathcal{A}}) + p^i)); e(x^i_{\mathcal{A}})) - \gamma \|p^i\|_1 - \omega g(h(e(x^i_{\mathcal{A}}) + p^i)\right]  $$  
      Gradient ascent:  
      $$ p^i \leftarrow p^i + ps \cdot \psi^i $$  
     **end for**

     $\gamma \leftarrow \min\left( \max\left( \gamma - \beta\left( \rho - \frac{1}{|T_B|} \sum_{i=1}^{|T_B|} \|p^i\| \right), 0 \right), \gamma_{\max} \right)$
    $\mathbf{w} \leftarrow \mathbf{w} - \text{lr} \cdot \nabla_{\mathbf{w}} \left[
    \frac{1}{|T_B|} \sum_{i=1}^{|T_B|} \ell_{\mathrm{OE}}(h(g(x^i_{\mathcal{A}}) + p^i)))- \omega g(h(e(x^i_{\mathcal{A}}) + p^i)) + 
    \alpha \cdot \frac{1}{|S_B|} \sum_{i=1}^{|S_B|} \ell(f_{\mathbf{w}}(x^i_1), y^i_1)
    \right]$

    

    

    **end for**

    **Output**: Final model parameter  $\mathbf{w}$ 

Ideal idea is to add a penalty or an repulsion to original (based on distribution), but not easy to perfectly fit in each dual process. So considering from the end-point and then try to derive an 



| Epoch | ID Acc (%) | FPR@95%TPR | AUROC | AUPR  |
| ----- | ---------- | ---------- | ----- | ----- |
| 10    | 84.84      | 15.20      | 95.89 | 99.02 |
| 20    | 89.51      | 4.88       | 98.72 | 99.71 |
| 30    | 91.98      | 6.55       | 98.11 | 99.56 |
| 40    | 94.18      | 3.65       | 98.74 | 99.71 |
| 50    | 95.12      | 1.97       | 99.07 | 99.82 |


| Epoch | ID Acc (%) | FPR@95%TPR | AUROC | AUPR  |
| ----- | ---------- | ---------- | ----- | ----- |
| 10    | 84.68      | 12.00      | 96.60 | 99.19 |
| 20    | 88.51      | 3.50       | 98.86 | 99.74 |
| 30    | 90.84      | 5.97       | 98.10 | 99.56 |
| 40    | 94.02      | 3.18       | 98.73 | 99.74 |
| 50    | 94.18      | 1.88       | 99.11 | 99.83 |