# KFGOD

Yongxiang Li, Xinyu Su, **Zhong Yuan***, Run Ye, Dezhong Peng, and Hongmei Chen, A Kernelized Fuzzy Approximation Fusion Model with Granular-ball Computing for Outlier Detection, Information Fusion, 2026, 127: 103716.

## Abstract
Outlier detection is a fundamental task in data analytics, where fuzzy rough set-based methods have gained increasing attention for their ability to effectively model uncertainty associated with outliers in data. However, existing FRS-based methods often exhibit limitations when applied to complex scenarios. Most of these methods rely on single-granularity fusion, where all samples are processed at a uniform, fine-grained level. This restricts their ability to fuse multi-granularity information, limiting outlier discrimination and making them more susceptible to noise. Moreover, many traditional methods construct fuzzy relation matrices under linear assumptions, which fail to effectively represent the intricate, nonlinear relations commonly found in real-world data. This leads to suboptimal estimation of membership degrees and degrades the reliability of outlier detection. To address these challenges, we propose a Kernelized Fuzzy approximation fusion model with Granular-ball computing for Outlier Detection (KFGOD), which integrates multi-granularity granular-balls and kernelized fuzzy rough sets into a unified framework. KFGOD fuses multi-granularity information to capture abnormal information at different granularity levels. Simultaneously, kernel functions are employed to effectively model multi-granularity nonlinear relations, enhancing the expressive power of fuzzy relation construction. By performing information fusion across multiple kernelized fuzzy information granules associated with each granular-ball, KFGOD evaluates the outlier degrees of each ball and propagates this fused abnormality information to the corresponding samples. This hierarchical and kernelized method allows for effective outlier detection in unlabeled datasets. Extensive experiments conducted on twenty benchmark datasets confirm the effectiveness of KFGOD, which consistently outperforms several state-of-the-art baselines in terms of detection accuracy and robustness.

## Framework
![image](./paper/KFGOD_Framework.png)

## Directory Structure
```
.
├── code/                                 # Source code
│   └── GB_generation_with_idx.py         # Granular-ball generation
│   └── KFGOD.py                          # Main entry point
│   └── Example.csv                       # Example dataset
├── datasets/                             # Benchmark datasets used in the paper
├── paper/                                # Original paper
│   └── 2025_INF__A_Kernelized_Fuzzy_Approximation_Fusion_Model_with_Granular_ball_Computing_for_Outlier_Detection.pdf
│   └── KFGOD_Framework.png
└── README.md                             # Project readme
```

## Usage

### 1. Install dependencies
Ensure Python 3.8+ is installed. Required packages include:

- numpy  
- scikit-learn  

Install with:

```bash
pip install numpy scikit-learn
```

### 2. Run the code
```bash
cd code
python KFGOD.py
```

You can run KFGOD.py:
```python
if __name__ == "__main__":
    data = pd.read_csv("./Example.csv").values
    X = data[:, :-1]
    n, m = X.shape
    labels = data[:, -1]
    ID = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        scaler = MinMaxScaler()
        X[:, ID] = scaler.fit_transform(X[:, ID])

    GBs = get_GB(X)
    n_gb = len(GBs)
    print(f"The number of Granular-ball: {n_gb}")
    
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:,:-1], axis=0)
        
    delta = 0.3
    OD_gb = KFGOD(centers, delta)
    
    '''Map to samples'''
    OD = np.zeros(n)
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:,-1].astype('int')
        OD[point_idxs] = OD_gb[idx]
    print(OD_gb)
```
You can get outputs as follows:
```
The number of Granular-ball: 14
[0.10871106 0.06228779 0.09500879 0.04224544 0.10286076 0.07607004
 0.11049506 0.07432416 0.06844139 0.0980779  0.12767074 0.05458831
 0.08963456 0.03849735]
```
## Citation
If you find KFGOD useful in your research, please consider citing:
```
@article{li2025kfgod,
  title={A Kernelized Fuzzy Approximation Fusion Model with Granular-ball Computing for Outlier Detection},
  author={Li, Yongxiang and Su, Xinyu and Yuan, Zhong and Ye, Run and Peng, Dezhong and Chen, Hongmei},
  journal={Information Fusion},
  volume = {127},
  year={2026},
  pages = {103716}
}
```
## Contact
If you have any questions, please contact rhythmli.scu@gmail.com or yuanzhong@scu.edu.cn.
