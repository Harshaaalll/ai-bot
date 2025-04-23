# 🚗 EV Simulink Model + 📚 LLM-powered PDF QA App

This repository contains two projects:

1. **Electric Vehicle Simulink Model (MATLAB)**
2. **Flask-based LLM PDF QA System (Python)**

---

## 🚗 1. EV Simulink Model (MATLAB)

### Overview
Simulates an Electric Vehicle using MATLAB/Simulink with initialization from a script file.

### Files
- `ev_new.slx` - Simulink model of the EV.
- `init_ev_model.m` - Script to initialize simulation parameters.

### How to Run (MATLAB or MATLAB Online)
1. Rename `init_ev_model.m.txt` to `init_ev_model.m`.
2. Upload both files to MATLAB or [MATLAB Online](https://matlab.mathworks.com).
3. In the MATLAB command window, run:
   ```matlab
   init_ev_model
   open_system('ev_new')
   sim('ev_new')
