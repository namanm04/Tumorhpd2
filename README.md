# Tumorhpd2

**Tumorhpd2**: Prediction and Designing of Tumor Homing Peptides  

---

## Overview

**Tumorhpd2** is a computational tool and web server designed for researchers working with tumor homing peptides (THPs). Tumor homing peptides have the ability to recognize and bind specifically to tumor cells or tissues. These peptides are valuable for:

- Delivering target-specific drugs  
- Acting as imaging agents for therapeutics and diagnostics  

Accurate prediction and design of tumor homing peptides can significantly aid in effective cancer treatment management.

---

## Major Modules

### 1. Prediction

This module allows users to **predict whether a given peptide is tumor homing or non-homing** using composition-based machine learning methods.  

**Instructions:**
- Provide the peptide sequence.
- Select the desired **prediction model**.  

**Prediction Models:**

| Dataset                   | Available Models          |
|----------------------------|--------------------------|
| Whole length dataset       | RF, Hybrid (RF + MERCI) |
| 5–10 length dataset        | ET, Hybrid (ESM + MERCI) |

---

### 2. Design

This module facilitates **designing tumor homing peptides** by generating all possible single-residue mutants of a given peptide.  

**Features:**
- Generate all single-residue mutants.
- Predict whether each mutant is tumor homing or non-homing.

---

## Datasets

- **Whole length dataset**: Used for full-length peptide predictions.  
- **5–10 length dataset**: Used for short peptide predictions.  

Each dataset has two associated prediction models, as described above.

---

## Reference

Tumorhpd2 is developed for the scientific community to support research in targeted therapeutics and diagnostics using tumor homing peptides.

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/namanm04/Tumorhpd2.git
