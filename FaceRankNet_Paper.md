# FaceRankNet: Feature-Decomposed Ranking Framework for Explainable Facial Beauty Prediction

**Lucky Wijaya**
Computer Science Department, School of Computer Science
Bina Nusantara University, Jakarta, Indonesia 11480
lucky.wijaya@binus.ac.id

**Christian Rivaldi**
Computer Science Department, School of Computer Science
Bina Nusantara University, Jakarta, Indonesia 11480
christian.rivaldi@binus.ac.id

**Gregorius Willson**
Computer Science Department, School of Computer Science
Bina Nusantara University, Jakarta, Indonesia 11480
gregorius.willson@binus.ac.id

**Dr. Meiliana, S.Kom., M.Sc.**
Computer Science Department, School of Computer Science
Bina Nusantara University, Jakarta, Indonesia 11480
meiliana002@binus.ac.id

**Fiqri Ramadhan Tambunan, S.Kom., M.Kom.**
Computer Science Department, School of Computer Science
Bina Nusantara University, Jakarta, Indonesia 11480
fiqri.tambunan001@binus.ac.id

---

## Abstract

Facial Beauty Prediction (FBP) systems have achieved high accuracy but remain hindered by their "black-box" nature and high vulnerability to demographic bias. Current architectures predominantly output a single holistic score, failing to articulate the aesthetic contributions of specific facial organs, a limitation severely exacerbated by the scarcity of part-level dataset annotations. To resolve these critical flaws, this paper proposes **FaceRankNet**, a novel feature-decomposed framework that relies entirely on a pure dense geometric approach. By strictly discarding pixel texture to structurally mitigate bias, the framework extracts 468 spatial 3D nodes mapped into distinct anatomical sub-graphs, which are subsequently processed by a part-based Graph Attention Network (GAT). To overcome the absence of local feature annotations, we introduce a Weakly Supervised Learning mechanism anchored by the computational Averageness hypothesis, generating deterministic pseudo-labels that enable feature-level pairwise ranking. Evaluated on the SCUT-FBP5500 benchmark, the proposed architecture successfully outputs quantifiable, mutually comparable attractiveness scores for individual facial components alongside a global rating. FaceRankNet shifts the FBP paradigm from holistic approximation to granular, feature-decomposed assessment, presenting a comprehensive solution for fair and highly interpretable facial aesthetic evaluation.

**Keywords:** facial beauty prediction, explainable AI, graph attention network, multi-task learning

---

## I. Introduction

The perception of human facial attractiveness is a fundamental component of social cognition, profoundly influencing human interactions, psychological well-being, and evolutionary biology [1]. Driven by the proliferation of deep learning frameworks and large-scale benchmark datasets such as SCUT-FBP5500 [2], FBP systems have evolved from utilizing basic heuristic rules to deploying complex Convolutional Neural Networks (CNNs).

Despite these impressive accuracy levels, current FBP architectures suffer from two critical, intersecting flaws that hinder their practical and ethical deployment: an absolute lack of feature-level explainability and a high vulnerability to demographic bias [3].

The first major limitation is the "black-box" nature of modern FBP regressors. Current models evaluate faces holistically, outputting a single aggregate scalar value without clarifying the aesthetic contributions of specific facial organs (e.g., the proportion of the nose or the symmetry of the eyes) [4][5]. Existing explainability often relies on post-hoc region attribution methods like Grad-CAM [6], which only provide coarse spatial heatmaps rather than quantifiable feature-level justifications. Resolving this is severely hindered by annotation scarcity; standard datasets only provide global scores, preventing direct supervised learning for part-based aesthetic evaluation [2][7].

To bridge this research gap, we propose **FaceRankNet**, a framework fundamentally designed to provide a high degree of explainability of each face feature. The novelty of this research encompasses three main aspects:

1. **High-Precision Geometric Decomposition:** Instead of usual cropping techniques, we propose an extreme expansion from traditional sparse landmarks to a 468-node spatial 3D topology via MediaPipe [8]. This allows the precise mathematical partitioning of the face into distinct organ sub-graphs without destroying their spatial harmony.

2. **Weakly Supervised Learning via "Averageness":** To overcome the absence of local feature annotations, we introduce Geometric Pseudo-Pairing. The framework utilizes the computational "Averageness" (Koinophilia) hypothesis [9], computing a dataset-derived "Universal Average Face" as an objective mathematical anchor to automatically generate localized pseudo-labels [10].

3. **Feature-Level Pairwise Ranking:** We extend the learning-to-rank paradigm [11] to the component level using a part-based Graph Attention Network (GAT) [12]. This enables the architecture to directly compare, order, and rank the desirability of specific facial features across different individuals.

---

## II. Related Work

This section reviews the current state of facial beauty prediction research, organized into five key areas: deep learning architectures for FBP, learning-to-rank approaches, multi-task and multi-stream methods, explainability and interpretability. We analyze the strengths and limitations of each area to motivate the proposed feature-decomposed ranking approach.

### A. Multi-Task and Multi-Stream Methods

Various multi-stream and multi-task learning (MTL) frameworks capture localized aesthetics by processing specific facial parts in parallel alongside the full-face image, ultimately aggregating them through late fusion [1]. These frameworks often jointly optimize auxiliary tasks such as age estimation and landmark detection to regularize predictions and improve generalization [13], with recent innovations like feature fusion Transformers to dynamically integrate global and local facial characteristics for more robust assessment [14].

While these methods successfully demonstrate that facial beauty comprises regional contributions, they process facial parts independently before fusion and fail to explicitly model geometric feature interactions. Consequently, the extracted per-stream features serve merely as intermediate representations rather than providing directly interpretable, feature-level beauty scores.

### B. Geometric and Graph Modeling Integration

To explicitly model the structural interactions that multi-stream CNNs miss, architectures like DeepGeoFusion construct anatomically constrained facial graphs from 86 landmarks via Delaunay triangulation to align global visual features [15], while hybrid models like GeoFusion-Net deploy Graph Attention Networks (GAT) to learn relational features such as proportionality and symmetry [16]. Meanwhile, the GCF framework demonstrates that representing extracted visual features on a graph and enhancing them with graph convolutional layers can significantly refine the capture of nuanced facial expressions [17].

However, relying on visual-geometric fusion leaves networks vulnerable to demographic biases, and as emphasized by Ye et al. [18], traditional sparse landmarks are anatomically insufficient to precisely capture nuanced shape differences and smoothly decompose complex organs into independent sub-graphs [19][20]. To bridge this gap, FaceRankNet abandons visual fusion entirely, instead ensuring structurally-driven aesthetic evaluation by utilizing a pure, dense geometric topology expanded to 468 spatial 3D nodes [8][21] that are processed directly by a part-based Graph Attention Network [12][22].

### C. Weakly Supervised Learning and the Averageness Hypothesis

The critical bottleneck in component-level beauty evaluation is annotation scarcity, making fully supervised local feature training impossible. As collecting dense, part-level annotations is prohibitively expensive and highly subjective, Weakly Supervised Learning (WSL) has become an essential paradigm [23]. While recent advancements demonstrate that WSL can successfully isolate local facial components using only global labels [24], unconstrained pseudo-labeling poses significant risks of semantic collapse and noise amplification when applied to beauty prediction [25].

To bypass this vulnerability, FaceRankNet introduces a highly constrained, deterministic WSL mechanism anchored by the Koinophilia (Averageness) hypothesis. Initially popularized by Langlois and Roggman [9] and extensively analyzed in recent reviews [1], this hypothesis posits that faces with structural proportions closely resembling the population mean are universally perceived as more attractive, as mathematical averaging effectively neutralizes extreme structural distortions and spatial asymmetries [1][26].

FaceRankNet uniquely operationalizes this psychological theory as a computational Geometric Pseudo-Pairing mechanism. By computing the "Universal Average Face" from the dataset to serve as an objective geometric baseline, the framework automatically generates deterministic local pseudo-labels. This approach successfully executes weakly supervised learning for part-based features while completely avoiding the instability of unconstrained WSL.

### D. Learning-to-Rank in Facial Beauty Prediction

Human beauty assessment is rarely an absolute scalar valuation; rather, it is intrinsically comparative [3]. Cognitive psychology dictates that humans naturally perceive attractiveness through relative comparisons — subconsciously evaluating whether one individual possesses more symmetrical eyes or a more proportionate jawline than another.

To computationally mimic this cognitive behavior, learning-to-rank frameworks have emerged as a powerful alternative to standard regression. Notably, the R³CNN framework [11] pioneered this by deploying Siamese architectures to train models to order pairs of faces. However, a critical blind spot persists in modern FBP literature: current ranking methodologies are severely restricted to macro-level, holistic evaluations [7]. They treat the face as an indivisible entity, failing to articulate why Face A is predicted to be more attractive than Face B at the feature level [27].

FaceRankNet addresses this limitation by extending the pairwise ranking paradigm directly to individual facial features. By coupling the ranking loss with the objective pseudo-labels generated in the weakly supervised module, the framework can explicitly evaluate and order specific components (e.g., comparing the structural proportions of Nose A directly against Nose B). This straightforward extension successfully bridges the gap between holistic beauty prediction and true part-based explainability in FBP.

---

## III. Methodology

The FaceRankNet framework is a pure dense geometric system designed to produce fair and explainable facial beauty scores without relying on pixel texture or skin tone information. By operating exclusively on 3D geometric coordinates extracted from facial landmarks, the architecture structurally eliminates demographic bias at the input level [3].

> **Fig. 1.** FaceRankNet System Pipeline

### A. Dataset

The primary dataset used in this study is **SCUT-FBP5500** [2], a widely adopted benchmark for facial beauty prediction research. The dataset comprises 5,500 frontal facial images annotated with holistic beauty scores ranging from 1 to 5 by 60 human raters per image, resulting in reliable averaged ground-truth labels. The dataset encompasses diverse subjects across Asian and Caucasian ethnicities, both male and female, photographed under varied conditions. The standard train/test split of 80%/20% (4,400 training images and 1,100 test images) is adopted in accordance with established benchmarks.

It is important to note that SCUT-FBP5500 contains only holistic-level beauty annotations, directly motivating the weakly supervised learning mechanism used in this framework [19].

### B. Data Preprocessing and Geometric Extraction

Prior to model input, each facial image undergoes a standardized geometric preprocessing pipeline to ensure consistency and scale invariance across the dataset.

1. **Face Detection and Alignment:** Each image is processed through the MediaPipe Face Mesh algorithm to detect and extract 468 three-dimensional spatial landmark coordinates $(X_i, Y_i, Z_i)$ per face [8]. MediaPipe assigns consistent anatomical indices across all faces, ensuring reproducible landmark-to-anatomy mapping.

2. **Coordinate Normalization:** Raw landmark coordinates are normalized through Centroid Centering (subtracting the geometric center from each coordinate) and Inter-ocular Scale Normalization (dividing all coordinates by the Euclidean distance between the outer corners of the eyes) [10]. This scale normalization makes the geometric representation invariant to subject distance from the camera. After normalization, each face is represented as a matrix $F \in \mathbb{R}^{468 \times 3}$.

3. **Sub-Graph Partitioning:** Landmarks are partitioned into five anatomically defined sub-graphs based on the canonical MediaPipe index mapping: Left Eye + Left Brow, Right Eye + Right Brow, Nose, Lips/Mouth, and Jawline/Face Shape. Eyebrow nodes are intentionally merged into their respective eye sub-graphs to ensure the attention mechanism captures inter-landmark relationships within the brow-eye complex [5].

### C. Model Architecture

> **Fig. 2.** Data Preprocessing and Model Architecture WorkFlow

FaceRankNet processes each face as a set of five parallel sub-graphs, each independently evaluated by a dedicated Graph Attention Network (GAT), before fusion into a global aesthetic score.

1. **Node Embedding:** Each node's normalized coordinates are projected into a higher-dimensional embedding space via a shared linear transformation.

2. **Graph Attention Network (GAT):** Each sub-graph is processed by an independent GAT layer. The attention coefficient $\alpha_{ij}$ is computed as:

$$e_{ij} = \text{LeakyReLU}(a^T [Wh_i \| Wh_j])$$

$$\alpha_{ij} = \text{softmax}_j(e_{ij})$$

Each node's representation is then updated by aggregating from its neighbors weighted by these attention scores:

$$h'_i = \sigma\!\left(\sum_j \alpha_{ij} \cdot W \cdot h_j\right)$$

3. **Sub-Graph Pooling:** Node representations within a sub-graph are aggregated into a single organ-level vector via attention pooling:

$$h_{\text{organ}} = \sum_i \beta_i \cdot h'_i, \quad \beta_i = \text{softmax}(w^T h'_i)$$

This allows the model to identify which specific landmarks are most informative for each organ's aesthetic evaluation.

4. **Local Score Prediction:** The pooled representation vector $h_{\text{organ}}$ is passed through a Multi-Layer Perceptron (MLP) and constrained to the ordinal scale $[1, 5]$ via a scaled Sigmoid activation:

$$\hat{y}_{\text{organ}} = 4 \cdot \text{Sigmoid}(\text{MLP}(h_{\text{organ}})) + 1$$

5. **Global Score Fusion:** The global aesthetic score is computed as the learned weighted sum of the five local organ scores:

$$\hat{y}_{\text{global}} = \sum_i \text{softmax}(w_i) \cdot \hat{y}_i$$

The Softmax constraint ensures mathematically that $\hat{y}_{\text{global}} \in [1, 5]$. Crucially, the learned weights $w_i$ provide interpretability regarding the relative aesthetic importance of each facial organ.

### D. Model Training

> **Fig. 3.** FaceRankNet Training Workflow

The architecture is trained using a multi-task learning approach integrating pseudo-labels and a hybrid loss function.

#### 1. Weakly Supervised Pseudo-Label Generation

Based on the Averageness Hypothesis [9], a Universal Average Face is constructed by computing the coordinate-wise mean of all 468 normalized landmarks across the entire training set. For each facial organ sub-graph, the deviation of a given face from the average face is computed as the Mean Squared Error (MSE) of the organ's normalized coordinates:

$$MSE_{\text{organ}} = \frac{1}{N_{\text{organ}}} \sum_i \|p_i - \mu_i\|^2$$

where $p_i$ is the normalized coordinate of node $i$ and $\mu_i$ is the corresponding coordinate in the Universal Average Face. This MSE is then converted to a continuous pseudo-score on the $[1, 5]$ scale via linear mapping:

$$\hat{y}^{psc}_{\text{organ}} = 5 - 4 \cdot \left(\frac{MSE_{\text{organ}}}{\max(MSE_{\text{organ}})}\right)$$

#### 2. Pairwise Training Construction

During training, face pairs $(A, B)$ are sampled from the training set. For each pair, pseudo-scores $\hat{y}^{psc}_{\text{organ}}(A)$ and $\hat{y}^{psc}_{\text{organ}}(B)$ determine the ranking direction per organ. Specifically, if $\hat{y}^{psc}_{\text{organ}}(A) > \hat{y}^{psc}_{\text{organ}}(B)$, the model is penalized if its predicted local score for face $A$'s organ is not greater than that for face $B$. This extends the pairwise ranking paradigm from R³CNN [11], which operates at the holistic level, to the individual feature level.

#### 3. Hybrid Loss Function

The model is jointly optimized using three simultaneous loss components:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{rank}} + \lambda_3 \mathcal{L}_{\text{div}}$$

**a. Anchor Regression Loss:** Mean Squared Error against the SCUT-FBP5500 ground-truth holistic label:

$$\mathcal{L}_{\text{reg}} = \frac{1}{N} \sum (\hat{y}_{\text{global}} - y_{gt})^2$$

**b. Feature-Level Ranking Loss:** A pairwise ranking penalty applied at the organ level:

$$\mathcal{L}_{\text{rank}} = \sum_{\text{organ}} \sum_{(A,B)} \log(1 + \exp(\hat{y}_{\text{organ}}(B) - \hat{y}_{\text{organ}}(A)))$$

where $(A, B)$ are pairs where face $A$ has a higher pseudo-score than face $B$ for that organ.

**c. Diversity Regularization:** A variance penalty to prevent uniform score across all organs:

$$\mathcal{L}_{\text{div}} = -\text{Var}(\hat{y}_{\text{organ}})$$

#### 4. Backpropagation and Dynamic Gradient Normalization (GradNorm)

The computed total loss ($\mathcal{L}_{\text{total}}$) serves as the singular objective function driving the backpropagation process to update the learnable parameters across the Graph Attention Networks and MLPs. However, because the regression task ($\mathcal{L}_{\text{reg}}$) operates on human-annotated labels while the ranking tasks ($\mathcal{L}_{\text{rank}}$) operate on generated pseudo-labels, their gradient magnitudes differ substantially. To prevent the global regression task from dominating the backpropagation phase and suppressing the local ranking objectives, **GradNorm** is deployed. GradNorm dynamically recalibrates the loss weights ($\lambda_i$) at each backward pass to equalize gradient magnitudes, guaranteeing synchronous convergence across all multi-task branches without requiring manual hyperparameter tuning [28].

### E. Evaluation

To rigorously quantify the model's performance on the testing set, the framework is evaluated across three primary dimensions: predictive accuracy, demographic fairness, and feature-level explainability.

**Global Score Prediction Accuracy** is benchmarked using the Pearson Correlation Coefficient (PCC) to measure linear correlation, complemented by the Mean Absolute Error (MAE) to capture absolute prediction magnitude errors:

$$PCC = \frac{\text{Cov}(\hat{y}, y)}{\sigma_{\hat{y}} \cdot \sigma_y}, \quad MAE = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

**Demographic Fairness** is validated using the Demographic Parity Difference (DPD), evaluating the disparity in prediction errors between Asian and Caucasian sub-groups [3]:

$$DPD = |MAE_{\text{Asian}} - MAE_{\text{Caucasian}}|$$

A DPD score approaching zero indicates equitable performance, proving the model does not disproportionately penalize specific demographic groups.

**Feature-Level Explainability** is evaluated through the model's part-based outputs:

- **Local Score Validity:** Assesses the generated organ scores ($\hat{y}_{\text{organ}}$) to ensure semantic fidelity, verifying that facial components geometrically closer to the Universal Average Face receive proportionally higher aesthetic scores.
- **Organ Importance Distribution:** Analyzes the Softmax distribution of the dynamically learned global fusion weights ($w_i$) to statistically quantify the relative aesthetic contribution of each specific facial organ to the overall beauty prediction.

---

## IV. Experimental Design

The experiment is structured to evaluate the FaceRankNet framework's ability to provide explainable beauty prediction while mitigating demographic bias compared to holistic baselines.

### A. Setup and Implementation Details

The SCUT-FBP5500 dataset is divided into 80% training (4,400 images) and 20% testing (1,100 images). Preprocessing involves extracting 468 3D landmarks via MediaPipe, applying Centroid Centering, and Inter-ocular Scale Normalization. The sub-graphs represent the eyes, nose, mouth, and jawline. The model is implemented in PyTorch utilizing Deep Graph Library (DGL) for the Graph Attention Networks.

---

## V. Conclusion

FaceRankNet presents a paradigm shift in facial beauty prediction by abandoning black-box scoring for a feature-decomposed scoring and more explainable architecture. By …

---

## Author Contribution

- **Lucky Wijaya:** Methodology, Formal Analysis, Writing, Review & Editing.
- **Gregorius Willson:** Conceptualization (Problem Identification), Methodology (Methodological Conceptualization), Writing.
- **Christian Rivaldi:** Literature Review, Writing, Editing.
- **Meiliana:** Manuscript Review and Topic Identification.
- **Fiqri Ramadhan:** Guiding Paper Creation and Manuscript Review.

---

## References

[1] A. Ibrahem, J. Saeed, and A. M. Abdulazeez, "Insights into Automated Attractiveness Evaluation from 2D Facial Images: A Comprehensive Review," *The International Arab Journal of Information Technology*, vol. 22, no. 1, pp. 77–98, Jan. 2025.

[2] L. Liang, L. Lin, L. Jin, D. Xie, and M. Li, "SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction," in *Proc. 24th Int. Conf. Pattern Recognit. (ICPR)*, Beijing, 2018, pp. 1598–1603.

[3] T. Iyer et al., "Machine Learning-based Facial Beauty Prediction and Analysis of Frontal Facial Images Using Facial Landmarks and Traditional Image Descriptors," *Computational Intelligence and Neuroscience*, vol. 2021, pp. 1–14, 2021.

[4] J. Zhao, M. Wu, L. Zhou, X. Wang, and J. Jia, "Cognitive Psychology-based Artificial Intelligence Review," *Frontiers in Neuroscience*, vol. 16, pp. 1–9, 2022.

[5] H. Ren, X. Chen, and Y. Zhang, "Correlation between Facial Attractiveness and Facial Components Assessed by Laypersons and Orthodontists," *Journal of Dental Sciences*, vol. 16, no. 1, pp. 431–436, 2021.

[6] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.

[7] N. Weng, J. Wang, A. Li, and Y. Wang, "Two-Stream Temporal Convolutional Network for Dynamic Facial Attractiveness Prediction," in *Proc. 25th Int. Conf. Pattern Recognit. (ICPR)*, Milan, 2021, pp. 10026–10033.

[8] C. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," in *Proc. CVPR Workshops*, 2019, pp. 1–9.

[9] J. Langlois and L. Roggman, "Attractive Faces are only Average," *Psychological Science*, vol. 1, no. 2, pp. 115–121, 1990.

[10] D. Zhang, Q. Zhao, and F. Chen, "Quantitative Analysis of Human Facial Beauty Using Geometric Features," *Pattern Recognition*, vol. 44, no. 4, pp. 940–950, 2011.

[11] L. Lin, L. Liang, and L. Jin, "Regression Guided by Relative Ranking Using Convolutional Neural Network (R³CNN) for Facial Beauty Prediction," *IEEE Transactions on Affective Computing*, vol. 13, no. 1, pp. 122–134, 2022.

[12] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph Attention Networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.

[13] A. H. Ibrahem and A. M. Abdulazeez, "A Comprehensive Review of Facial Beauty Prediction Using Multi-Task Learning and Facial Attributes," *ARO-The Scientific Journal of Koya University*, vol. 13, no. 1, pp. 10–25, 2025.

[14] J. Gan, L. Li, and S. Zhao, "Global-Local Feature Fusion With Transformer for Facial Beauty Prediction," *IEEE Transactions on Computational Social Systems*, vol. 11, no. 1, pp. 1125–1136, Feb. 2024.

[15] K. Wang, Y. Li, D. Huang, J. Feng, and X. Feng, "DeepGeoFusion: Personalized facial beauty prediction through geometric-visual fusion," *Frontiers in Computer Science*, vol. 7, p. 1692523, 2026.

[16] X. Li, Q. Zhao, and F. Chen, "GeoFusion-Net: A Hybrid Graph and Convolutional Network for Facial Beauty Prediction," *IEEE Transactions on Affective Computing* (Preprint), pp. 1–12, 2025.

[17] H. Kassab, M. Bahaa, and A. Hamdi, "GCF: Graph Convolutional Networks for Facial Expression Recognition," arXiv preprint arXiv:2407.02361v1, 2024.

[18] Y. Ye, G. Yan, D. Wen, and M. Tan, "Optimized facial landmark modeling with medical aesthetic constraints by a multi-objective genetic algorithm," *Frontiers in Computational Neuroscience*, vol. 20, p. 1705259, 2026, doi: 10.3389/fncom.2026.1705259.

[19] S. Tong, X. Liang, T. Kumada, and S. Iwaki, "Putative Ratios of Facial Attractiveness in a Deep Neural Network," *Vision Research*, vol. 178, pp. 86–99, 2021, doi: 10.1016/j.visres.2020.10.003.

[20] W. Zheng et al., "3D Dense Face Alignment via Graph Convolutional Networks," *Pattern Recognition*, vol. 124, 2022.

[21] W. Rahman et al., "Real-Time Face Age Detection System Based on Deep Neural Networks with MediaPipe Optimization for Enhanced Accuracy," *APIC*, 2024.

[22] A. K. Singh et al., "Region-wise landmarks-based feature extraction employing SIFT, SURF, and ORB feature descriptors to recognize Monozygotic twins from 2D/3D Facial Images," *F1000Research*, 2024.

[23] S. Zhao et al., "Weakly Supervised Learning for Facial Affective Behavior Analysis: A Review," arXiv preprint arXiv:2101.09858, 2021.

[24] X. Wang et al., "DisFaceRep: Representation Disentanglement for Co-occurring Facial Components in Weakly Supervised Face Parsing," in *Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)*, Dublin, Ireland, 2025.

[25] D. Zhu, X. Shen, M. Mosbach, A. Stephan, and D. Klakow, "Weaker Than You Think: A Critical Look at Weakly Supervised Learning," in *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, Toronto, Canada, 2023, pp. 14229–14253.

[26] K. Kleisner et al., "Distinctiveness and femininity, rather than symmetry and masculinity, affect facial attractiveness across the world," *Evolution and Human Behavior*, vol. 45, 2024.

[27] X. Yang et al., "A Ranking Information Based Network for Facial Beauty Prediction," *IEICE Transactions on Information and Systems*, vol. E107.D, no. 6, pp. 772–780, June 2024.

[28] Z. Chen, V. Badrinarayanan, C. Y. Lee, and A. Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks," in *Proceedings of the 35th International Conference on Machine Learning (ICML)*, 2018, pp. 794–803.
