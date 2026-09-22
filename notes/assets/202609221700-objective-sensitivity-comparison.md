## lrp-rli-gbl-012

### Fit quality (pooled out-of-fold)

| arm      | objective  | trees | leaves |    lr | OOF MAE | OOF RMSE | OOF R2 | OOF MedAE | in-sample R2 |
| :------- | :--------- | ----: | -----: | ----: | ------: | -------: | -----: | --------: | -----------: |
| registry | mae        |    11 |     19 | 0.165 |    6.15 |     9.45 |  0.578 |      3.41 |        0.828 |
| mae      | mae        |    19 |     24 |  0.09 |    6.37 |     9.96 |  0.531 |      3.38 |        0.738 |
| huber    | huber      |    54 |     40 |  0.05 |    6.06 |     8.77 |  0.637 |      3.68 |        0.934 |
| l2       | regression |    13 |     48 | 0.185 |    6.19 |     8.72 |   0.64 |      3.79 |        0.943 |
| poisson  | poisson    |    56 |     30 |   0.1 |    5.99 |     8.91 |  0.625 |      3.14 |        0.958 |

### Ranking agreement (permutation importance, vs the re-tuned MAE arm)

| arm      | top-5                                      | n z>=2 | replicated                       | rho vs mae | top-5 overlap | replicated overlap |
| :------- | :----------------------------------------- | -----: | :------------------------------- | ---------: | ------------: | :----------------- |
| registry | spphon, yarclet, eowpvt, aptgram, celf     |      9 | eowpvt, erbword, spphon, yarclet |       0.53 |             3 | 3/4                |
| mae      | spphon, erbword, yarclet, b1exto, eowpvt   |      7 | erbword, spphon, yarclet         |          1 |             5 | 3/3                |
| huber    | spphon, yarclet, erbword, aptgram, nonword |      9 | nonword, spphon, yarclet         |       0.65 |             3 | 2/4                |
| l2       | spphon, aptinfo, yarclet, nonword, erbword |      6 | spphon, yarclet                  |        0.4 |             3 | 2/3                |
| poisson  | spphon, yarclet, aptgram, nonword, erbword |      5 | spphon, yarclet                  |       0.58 |             3 | 2/3                |

### Rank, permutation z and SHAP direction for every predictor that is top-5 in any arm (~ = noisy or non-monotonic)

| feature | registry     | mae          | huber       | l2           | poisson     |
| :------ | :----------- | :----------- | :---------- | :----------- | :---------- |
| spphon  | #1 z=5.1 +   | #1 z=4.3 +   | #1 z=6.2 +  | #1 z=6.4 +   | #1 z=4.9 +  |
| yarclet | #2 z=4.3 +   | #3 z=3.4 +   | #2 z=3.5 +  | #3 z=3.7 +   | #2 z=3.7 +  |
| eowpvt  | #3 z=2.4 +   | #5 z=2.3 +   | #6 z=2.2 +  | #6 z=2.2 +   | #6 z=1.5 +  |
| aptgram | #4 z=2.9 +   | #8 z=1.8 +   | #4 z=3.9 +~ | #8 z=1.4 -~  | #3 z=3.7 +~ |
| celf    | #5 z=2.3 +~  | #31 z=-0.7 + | #11 z=0.8 + | #15 z=0.2 +  | #9 z=1.2 +  |
| erbword | #6 z=2.4 +   | #2 z=3.7 +   | #3 z=3.5 +~ | #5 z=3.0 +~  | #5 z=2.2 +  |
| b1exto  | #12 z=1.9 +  | #4 z=2.8 +   | #10 z=1.6 + | #29 z=-0.9 + | #7 z=1.5 +  |
| nonword | #7 z=4.2 +~  | #6 z=2.5 +   | #5 z=3.6 +  | #4 z=4.5 +   | #4 z=2.6 +  |
| aptinfo | #19 z=0.1 +~ | #7 z=4.2 +~  | #7 z=2.4 +  | #2 z=3.3 +~  | #10 z=1.5 + |

## lrp-rli-gbl-013

### Fit quality (pooled out-of-fold)

| arm      | objective  | trees | leaves |    lr | OOF MAE | OOF RMSE | OOF R2 | OOF MedAE | in-sample R2 |
| :------- | :--------- | ----: | -----: | ----: | ------: | -------: | -----: | --------: | -----------: |
| registry | mae        |   118 |     17 | 0.048 |    0.86 |     1.37 |  0.436 |      0.46 |        0.689 |
| mae      | mae        |   200 |     40 | 0.023 |    0.85 |     1.35 |  0.456 |      0.38 |        0.678 |
| huber    | huber      |   139 |     47 | 0.033 |    0.91 |     1.31 |   0.49 |      0.57 |        0.799 |
| l2       | regression |   218 |     53 | 0.013 |    0.95 |     1.31 |  0.488 |      0.64 |        0.808 |
| poisson  | poisson    |   312 |     42 | 0.022 |    0.94 |      1.4 |  0.412 |      0.51 |        0.893 |

### Ranking agreement (permutation importance, vs the re-tuned MAE arm)

| arm      | top-5                                     | n z>=2 | replicated                      | rho vs mae | top-5 overlap | replicated overlap |
| :------- | :---------------------------------------- | -----: | :------------------------------ | ---------: | ------------: | :----------------- |
| registry | ewrswr, spphon, b1reto, rowpvt, yarclet   |      4 | ewrswr, spphon                  |       0.69 |             4 | 2/2                |
| mae      | spphon, ewrswr, b1reto, yarcsi, yarclet   |      3 | ewrswr, spphon                  |          1 |             5 | 2/2                |
| huber    | ewrswr, spphon, yarclet, b1reto, rowpvt   |      4 | ewrswr, spphon, yarclet         |       0.68 |             4 | 2/3                |
| l2       | ewrswr, spphon, yarclet, yarcsi, b1reto   |      5 | ewrswr, spphon, yarclet, yarcsi |       0.57 |             5 | 2/4                |
| poisson  | ewrswr, yarclet, spphon, blending, rowpvt |      2 | ewrswr, spphon                  |       0.55 |             3 | 2/2                |

### Rank, permutation z and SHAP direction for every predictor that is top-5 in any arm (~ = noisy or non-monotonic)

| feature  | registry     | mae          | huber       | l2          | poisson     |
| :------- | :----------- | :----------- | :---------- | :---------- | :---------- |
| ewrswr   | #1 z=3.0 +   | #2 z=3.0 +   | #1 z=4.1 +  | #1 z=4.3 +  | #1 z=4.4 +  |
| spphon   | #2 z=2.7 +   | #1 z=3.2 +   | #2 z=3.1 +  | #2 z=3.8 +  | #3 z=2.7 +  |
| b1reto   | #3 z=2.2 +   | #3 z=2.3 +   | #4 z=2.3 -~ | #5 z=2.4 -~ | #6 z=1.5 -~ |
| rowpvt   | #4 z=1.4 +~  | #6 z=1.2 +~  | #5 z=1.9 +~ | #9 z=0.7 +~ | #5 z=1.3 +~ |
| yarclet  | #5 z=1.5 +   | #5 z=1.0 +   | #3 z=2.4 +  | #3 z=2.1 +  | #2 z=2.0 +  |
| yarcsi   | #6 z=1.1 +   | #4 z=1.6 +   | #7 z=1.2 +  | #4 z=2.0 +  | #7 z=0.4 +  |
| blending | #31 z=-1.2 + | #32 z=-2.0 + | #12 z=0.0 + | #8 z=0.9 +  | #4 z=1.5 +  |

## lrp-rli-gbl-006

### Fit quality (pooled out-of-fold)

| arm      | objective  | trees | leaves |    lr | OOF MAE | OOF RMSE | OOF R2 | OOF MedAE | in-sample R2 |
| :------- | :--------- | ----: | -----: | ----: | ------: | -------: | -----: | --------: | -----------: |
| registry | mae        |   103 |     20 | 0.096 |    6.07 |     7.66 |  0.709 |      5.15 |        0.959 |
| mae      | mae        |   562 |     59 | 0.011 |    5.96 |     7.58 |  0.715 |      5.03 |         0.94 |
| huber    | huber      |   103 |      8 | 0.037 |    5.93 |     7.49 |  0.722 |      5.06 |        0.945 |
| l2       | regression |   137 |      8 | 0.028 |    5.63 |     7.23 |  0.741 |      4.79 |        0.934 |
| poisson  | poisson    |   843 |     52 | 0.011 |    5.67 |     7.23 |  0.741 |      4.68 |        0.937 |

### Ranking agreement (permutation importance, vs the re-tuned MAE arm)

| arm      | top-5                                 | n z>=2 | replicated                                 | rho vs mae | top-5 overlap | replicated overlap |
| :------- | :------------------------------------ | -----: | :----------------------------------------- | ---------: | ------------: | :----------------- |
| registry | b1exto, aptinfo, rowpvt, celf, age    |      8 | age, aptinfo, b1exto, b1reto, celf, rowpvt |       0.81 |             4 | 4/7                |
| mae      | b1exto, aptinfo, rowpvt, celf, ewrswr |      7 | aptinfo, b1exto, celf, ewrswr, rowpvt      |          1 |             5 | 5/5                |
| huber    | b1exto, rowpvt, aptinfo, celf, age    |      6 | aptinfo, b1exto, celf, rowpvt              |       0.79 |             4 | 4/5                |
| l2       | b1exto, aptinfo, rowpvt, celf, age    |      6 | aptinfo, b1exto, celf, rowpvt              |       0.86 |             4 | 4/5                |
| poisson  | b1exto, aptinfo, rowpvt, celf, age    |      7 | aptinfo, b1exto, celf, rowpvt              |       0.91 |             4 | 4/5                |

### Rank, permutation z and SHAP direction for every predictor that is top-5 in any arm (~ = noisy or non-monotonic)

| feature | registry   | mae        | huber      | l2         | poisson    |
| :------ | :--------- | :--------- | :--------- | :--------- | :--------- |
| b1exto  | #1 z=4.3 + | #1 z=4.9 + | #1 z=5.3 + | #1 z=5.2 + | #1 z=4.9 + |
| aptinfo | #2 z=4.5 + | #2 z=4.7 + | #3 z=4.3 + | #2 z=4.3 + | #2 z=4.4 + |
| rowpvt  | #3 z=4.5 + | #3 z=4.7 + | #2 z=4.2 + | #3 z=4.8 + | #3 z=4.8 + |
| celf    | #4 z=4.0 + | #4 z=4.0 + | #4 z=3.0 + | #4 z=3.5 + | #4 z=3.4 + |
| age     | #5 z=3.9 + | #6 z=3.3 + | #5 z=2.9 + | #5 z=3.2 + | #5 z=3.4 + |
| ewrswr  | #6 z=1.7 + | #5 z=2.3 + | #6 z=1.7 + | #7 z=1.6 + | #6 z=1.6 + |

## lrp-rli-gbg-012

### Fit quality (pooled out-of-fold)

| arm      | objective  | trees | leaves |    lr | OOF MAE | OOF RMSE | OOF R2 | OOF MedAE | in-sample R2 |
| :------- | :--------- | ----: | -----: | ----: | ------: | -------: | -----: | --------: | -----------: |
| registry | mae        |   193 |     48 | 0.105 |    2.98 |     4.16 |  0.083 |      2.06 |          0.3 |
| mae      | mae        |   327 |     41 | 0.045 |    2.97 |     4.17 |  0.078 |      1.99 |        0.255 |
| huber    | huber      |    94 |     19 | 0.125 |    3.01 |     4.02 |   0.14 |      2.33 |        0.495 |
| l2       | regression |    43 |     16 |  0.09 |     3.1 |     4.12 |  0.099 |       2.4 |        0.492 |

### Ranking agreement (permutation importance, vs the re-tuned MAE arm)

| arm      | top-5                                 | n z>=2 | replicated   | rho vs mae | top-5 overlap | replicated overlap |
| :------- | :------------------------------------ | -----: | :----------- | ---------: | ------------: | :----------------- |
| registry | age, yarclet, b1exto, trog, hearing   |      2 | age, yarclet |       0.95 |             4 | 2/2                |
| mae      | age, yarclet, b1exto, trog, celf      |      2 | age, yarclet |          1 |             5 | 2/2                |
| huber    | age, eowpvt, yarclet, hearing, gender |      2 | age, eowpvt  |       0.66 |             2 | 1/3                |
| l2       | age, eowpvt, yarclet, trog, blending  |      2 | age, eowpvt  |       0.59 |             3 | 1/3                |

### Rank, permutation z and SHAP direction for every predictor that is top-5 in any arm (~ = noisy or non-monotonic)

| feature  | registry    | mae         | huber        | l2           |
| :------- | :---------- | :---------- | :----------- | :----------- |
| age      | #1 z=3.0 -  | #1 z=3.0 -  | #1 z=2.7 -   | #1 z=2.9 -   |
| yarclet  | #2 z=2.3 +  | #2 z=2.2 +  | #3 z=1.4 +   | #3 z=1.9 +   |
| b1exto   | #3 z=1.1 +  | #3 z=1.3 +  | #19 z=-0.0 + | #25 z=-0.4 + |
| trog     | #4 z=1.3 +  | #4 z=1.3 +  | #11 z=0.4 +  | #4 z=1.7 +   |
| hearing  | #5 z=1.6 +  | #7 z=1.0 +  | #4 z=1.3 +   | #6 z=1.1 +   |
| celf     | #6 z=0.7 +  | #5 z=0.6 +  | #10 z=0.3 +  | #26 z=-0.3 + |
| eowpvt   | #9 z=0.8 +  | #6 z=0.9 +  | #2 z=2.2 +   | #2 z=2.0 +   |
| gender   | #10 z=1.7 + | #10 z=1.5 + | #5 z=1.3 +   | #11 z=0.1 +  |
| blending | #12 z=0.6 + | #9 z=0.9 +  | #6 z=1.0 +~  | #5 z=1.4 +   |
