# Project Name : Development of deep ensembles for screening autism spectrum disorder and symptom severity using retinal photographs

- date : 2023-05-21 
- python version : 3.9.7

# Objective
- Create a deep ensemble model for screening autism spectrum disorder and symptom severity using retinal photographs
- Create PEPPR images to interpret the deep ensemble model

# Project Strudcture
- src : source code
- screening_ASD : Directory for screening autism spectrum disorder
- screening_symptom : Directory for Screening symptom severity

# Process
## Screening_ASD
```bash
cd screening_ASD && bash total_process.sh
```

## Screening_symptom
```bash
cd screening_symptom
bash process_ados.sh
bash process_srs.sh
```

run `3_bootstrap_and_figure.ipynb`

```
bash 4_OOD_inference.py
```

