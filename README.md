## Phonetic Analysis and Tempo Adjustments for Improved Speech Recognition in Dysarthric Speakers
Feifei Xiong · Jon Barker · Heidi Christensen IEEE ICASSP-2019

### Team
Saketh Reddy Vemula | 2022114014 | saketh.vemula@research.iiit.ac.in

Viswanath Vuppala | 2022101084 | viswanath.vuppala@research.iiit.ac.in

### File Structure
```
.
├── Analysis
│   ├── manner_of_articulation.png
│   ├── manner_of_articulation.py
│   ├── place_of_articulation.png
│   ├── place_of_articulation.py
│   ├── speaker_wise_MOA.png
│   ├── speaker_wise_MOA.py
│   ├── speaker_wise_POA.png
│   └── speaker_wise.py
├── ASR
│   ├── infer_wav2vec.py
│   ├── M01_results
│   │   ├── M01_Session1_wer_results.csv
│   │   └── M01_Session1_wer_summary.json
│   ├── M01_results_cleaned -> Contains results for Model-1
│   │   ├── M01_Session1_wer_results.csv
│   │   ├── M01_Session1_wer_results.xlsx
│   │   └── M01_Session1_wer_summary.json
│   ├── M01_results_model_2 -> Contains results for Model-2
│   │   ├── M01_Session1_wer_results.csv
│   │   ├── M01_Session1_wer_results.xlsx
│   │   └── M01_Session1_wer_summary.json
│   ├── M02_results
│   │   ├── M02_Session1_wer_results.csv
│   │   └── M02_Session1_wer_summary.json
│   ├── M02_results_cleaned -> Contains results for Model-1
│   │   ├── M02_Session1_wer_results.csv
│   │   ├── M02_Session1_wer_results.xlsx
│   │   └── M02_Session1_wer_summary.json
│   ├── M02_results_model_2 -> Contains results for Model-2
│   │   ├── M02_Session1_wer_results.csv
│   │   ├── M02_Session1_wer_results.xlsx
│   │   └── M02_Session1_wer_summary.json
│   ├── M04_results
│   │   ├── M04_Session2_wer_results.csv
│   │   └── M04_Session2_wer_summary.json
│   ├── M04_results_cleaned -> Contains results for Model-1
│   │   ├── M04_Session2_wer_results.csv
│   │   ├── M04_Session2_wer_results.xlsx
│   │   └── M04_Session2_wer_summary.json
│   ├── M04_results_model_2 -> Contains results for Model-2
│   │   ├── M04_Session2_wer_results.csv
│   │   ├── M04_Session2_wer_results.xlsx
│   │   └── M04_Session2_wer_summary.json
│   ├── overall_wer_summary.json -> manually compiled results of each models and Speaker combo
│   ├── phoneme_level_signal_domain.py
│   ├── SoX_phoneme_level.py -> SoX tempo adjustment for one sample
│   ├── SoX_phoneme_level_dataset.py -> SoX tempo adjustment for a Speaker
│   └── tempo_adjust_and_infer.py
├── Code
│   ├── energy_phonemes.py
│   ├── graphical_analysis_energy.py
│   ├── graphical_analysis.py
│   ├── graphical_analysis_scam.py
│   ├── phoneme_wise.py
│   ├── phonetic_analysis_energy.png
│   ├── phonetic_analysis.png
│   ├── phonetic_analysis.py
│   ├── phonetic_analysis_scam.png
│   ├── pitch_phonemes_2.py
│   ├── pitch_phonemes.py
│   └── rename.py
├── Dysarthria_Phonetic_Analysis(Durations).csv
├── Dysarthria_Phonetic_Analysis(Ratios).csv
├── Dysarthria_Phonetic_Analysis(Sheet1).csv
├── Dysarthria_Phonetic_Analysis(Sheet2 (2)).csv
├── README.md
├── Slides
│   ├── Final_Presentation.pdf
│   ├── project_mid.pdf
│   └── Seminar-2.pdf
├── tree.txt
└── utils
    ├── convert_csv_to_excel.py
    ├── extract_ratios_from_csv.py
    ├── M01_S1.json
    ├── M02_S1.json
    └── M04_S2.json

15 directories, 66 files

```
`ASR`: Contains scripts for inferencing Wav2Vec ASR model

`Code` and `Analysis`: Phonemic Analysis of Control and Dysarthric Speakers

`Slides`: Seminar-2, Mid Project and Final Project Slides for references. These slides contains links for excel sheets linked.

`utils`: helper codes for formatting, and changing formats of file.

