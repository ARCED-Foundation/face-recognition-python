# Face Recognition Script

A Python-based face recognition tool that compares faces from a CSV file against an image database using state-of-the-art AI models.

## Features

- 🔍 **Batch Processing** - Process multiple images from CSV files
- 📁 **Recursive Search** - Search through all subfolders automatically
- 🎯 **Multiple AI Models** - Choose from 9 different face recognition models
- 📊 **Excel Export** - Beautiful, color-coded Excel reports with summary statistics
- ⚡ **Fast Processing** - Efficient batch comparison with progress tracking
- 🎨 **Flexible Output** - CSV or formatted Excel output

## Installation

```bash
# Install required packages
pip install deepface pandas numpy opencv-python tf-keras openpyxl
```

**Get the script:**

Option 1: Clone the repository
```bash
git clone https://github.com/yourusername/face-recognition-python.git
cd face-recognition-python
```

Option 2: Download directly
- Download `face-recognize.py` from the repository
- Save it to your preferred directory

## Quick Start

### Basic Usage

```bash
python3 face-recognize.py -source data.csv -imageDB ./images -output results.csv
```

### With Excel Output

```bash
python3 face-recognize.py -source data.csv -imageDB ./images -output results.xlsx -excel
```

### Recursive Subfolder Search

```bash
python3 face-recognize.py -source data.csv -imageDB ./images -output results.xlsx -excel -recursive
```

**Note:** If the script is not in your current directory, include the full path:
```bash
python3 /path/to/face-recognize.py -source data.csv -imageDB ./images -output results.csv
```

## Input Format

Your CSV file must contain an `image` column with image filenames:

```csv
id,name,image,date
1,John Doe,person1.jpg,2024-01-15
2,Jane Smith,person2.jpg,2024-01-16
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-source` | Path to source CSV file | Required |
| `-imageDB` | Path to image database folder | Required |
| `-output` | Output file path (use `${today}` for date) | Required |
| `-excel` | Output as formatted Excel file | False |
| `-recursive` | Search subfolders recursively | False |
| `-model` | AI model to use (see comparison below) | Facenet |
| `-threshold` | Match threshold (0-100) | 70 |
| `-top` | Number of top matches to include | 5 |

**Note:** Use `${today}` in the output path to automatically insert today's date (format: YYYY-MM-DD).  
The script automatically replaces `${today}` with the current date when it runs.  
Example: `-output results-${today}.xlsx` becomes `results-2026-01-17.xlsx` (if run on January 17, 2026)

## AI Model Comparison

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| Facenet 👍 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose (Recommended) |
| ArcFace | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Highest accuracy, production systems |
| Facenet512 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Large databases, similar faces |
| OpenFace | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Real-time, speed priority |
| VGG-Face | ⭐⭐⭐⭐ | ⭐⭐ | Legacy systems |

### Model Details

- **Facenet** (Default): Best balance of speed and accuracy for most use cases
- **ArcFace**: State-of-the-art accuracy, ideal for production and security applications
- **Facenet512**: Higher accuracy than Facenet, better for distinguishing similar faces
- **OpenFace**: Fastest option, good for large datasets when speed matters
- **VGG-Face**: Classic model with good accuracy but slower processing

## Example Commands

### Basic comparison
```bash
python3 face-recognize.py -source data.csv -imageDB ./images -output results.csv
```

### High accuracy with ArcFace model
```bash
python3 face-recognize.py -source data.csv -imageDB ./images -output results.xlsx -excel -model ArcFace -threshold 75
```

### Fast processing with date in filename
```bash
python3 face-recognize.py -source data.csv -imageDB ./images -output ${today}-results.xlsx -excel -model OpenFace
```

### Full-featured example
```bash
python3 face-recognize.py \
  -source data.csv \
  -imageDB C:\Users\YourName\Documents\images \
  -output ${today}-results.xlsx \
  -excel \
  -recursive \
  -model ArcFace \
  -threshold 75 \
  -top 10
```

## Output

### CSV Output
Standard CSV file with match results and scores.

### Excel Output (`-excel` flag)
- **Color-coded status**: Green (success), Yellow (no face), Red (not found)
- **Score highlighting**: Green (80%+), Orange (60-79%), Red (<60%)
- **Auto-formatted columns** with proper widths
- **Summary sheet** with statistics
- **Frozen header row** for easy scrolling

### Output Columns
- Original CSV columns (preserved)
- `status`: Processing status
- `best_match`: Best matching image filename
- `best_match_score`: Similarity score (0-100)
- `total_comparisons`: Number of images compared
- `matches_found`: Number of matches above threshold
- `match_1_image`, `match_1_score`, etc.: Top N matches

## Requirements

- Python 3.7+
- Windows/Linux/macOS
- Supported image formats: JPG, JPEG, PNG, BMP, GIF

## Performance

Processing time varies by model and hardware:
- **OpenFace**: ~2-3 minutes for 100 images
- **Facenet**: ~3-4 minutes for 100 images
- **ArcFace**: ~5-8 minutes for 100 images
- **Facenet512**: ~5-7 minutes for 100 images

*Times based on CPU processing; GPU acceleration significantly faster*

## Troubleshooting

**Import Error**: Install all dependencies
```bash
pip install deepface pandas numpy opencv-python tf-keras openpyxl
```

**No face detected**: Ensure images contain clear, visible faces

**Slow processing**: Use OpenFace model or enable GPU acceleration

**Memory issues**: Process fewer images at a time or use a lighter model

## Acknowledgements

This project uses the following open-source libraries:

- **[DeepFace](https://github.com/serengil/deepface)** - A lightweight face recognition and facial attribute analysis framework for Python
  - License: MIT License
  - Created by Sefik Ilkin Serengil

- **Face Recognition Models**:
  - **Facenet** - Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
  - **ArcFace** - Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (2019)
  - **VGG-Face** - Parkhi et al., "Deep Face Recognition" (2015)
  - **OpenFace** - Amos et al., "OpenFace: A general-purpose face recognition library with mobile applications" (2016)

- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis library (BSD 3-Clause License)
- **[NumPy](https://numpy.org/)** - Numerical computing library (BSD License)
- **[OpenCV](https://opencv.org/)** - Computer vision library (Apache 2.0 License)
- **[openpyxl](https://openpyxl.readthedocs.io/)** - Excel file manipulation (MIT License)

## License

This script is provided as-is for educational and commercial use. Please ensure compliance with the licenses of all dependencies when using this tool.

## Support

For issues, questions, or contributions:
- **Email:** [mehrab.ali@arced.foundation](mailto:mehrab.ali@arced.foundation)
- **Organization:** [ARCED Foundation](https://arced.foundation)
- Open an issue on the repository

---

**Note**: Face recognition technology should be used responsibly and in compliance with local privacy laws and regulations.

---

## Author

**Mehrab Ali**  
Email: mehrab.ali@arced.foundation  
Organization: [ARCED Foundation](https://arced.foundation)  
Date: 17 January 2026

© 2026 Mehrab Ali, ARCED Foundation