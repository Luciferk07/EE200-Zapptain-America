import re

files = [
    r"c:\Users\karti\OneDrive\Desktop\EE200 Project\EE200_Project_Report.tex",
    r"c:\Users\karti\OneDrive\Desktop\EE200 Project\EE200 Q1 PDF\EE200_Project_Report.tex"
]

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find where Q3 starts
    q3_start = content.find("%====================================================\n\\section{Question 3: Sonic Signatures -- Audio Fingerprinting}")
    
    if q3_start != -1:
        # Keep everything before Q3
        new_content = content[:q3_start]
        
        # Add the Summary of Key Findings back, without Q3
        summary = """%====================================================
\\section{Summary of Key Findings}
%====================================================

\\begin{center}
\\begin{tabular}{>{}p{1.5cm} p{4.5cm} p{7cm}}
\\toprule
\\textbf{Q} & \\textbf{Core Technique} & \\textbf{Key Insight} \\\\
\\midrule
1A & 2D FFT + Notch Filter & Periodic noise $\\Rightarrow$ sparse spectral spikes. Gaussian notch beats binary. Radius tuning is critical. \\\\
\\addlinespace
1B & Sobel + Hysteresis & Differentiation amplifies noise: smooth first. Hysteresis preserves connected weak edges. \\\\
\\bottomrule
\\end{tabular}
\\end{center}

\\end{document}
"""
        new_content += summary
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Successfully trimmed {file_path}")
    else:
        print(f"Could not find Q3 section in {file_path}")
