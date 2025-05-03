# COMP841-001-Final-project-implementation

A deep learning project for classifying sickle cell images using a custom MobileNetV3 Small architecture. It features data augmentation, mixed precision training, and evaluation metrics like confusion matrices and ROC curves. Designed for efficiency, it is suitable for deployment in resource-constrained environments.

---

## **Instructions to Run the Project**

### **1. Clone the Repository**
- Open a terminal and run the following command:
  ```bash
  git clone https://github.com/your-username/your-repo-name.git
  ```
  Replace `your-username` and `your-repo-name` with the actual GitHub username and repository name.

### **2. Navigate to the Project Directory**
- Move into the project folder:
  ```bash
  cd Sickle-Classification
  ```

### **3. Set Up a Python Environment (Optional but Recommended)**
- Create a virtual environment to avoid conflicts with other Python packages:
  ```bash
  python3 -m venv env
  ```
- Activate the virtual environment:
  - On macOS/Linux:
    ```bash
    source env/bin/activate
    ```
  - On Windows:
    ```bash
    .\env\Scripts\activate
    ```

### **4. Install Dependencies**
- Install the required libraries using the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

### **5. Prepare the Dataset**
- Place the dataset in the `data_converted` folder (or update the `data_dir` variable in the code to point to your dataset location).

### **6. Run the Training Script**
- Start the training process by running the following command:
  ```bash
  python3 train_sickleCell.py
  ```

### **7. View Results**
- After training, the model's performance metrics (e.g., confusion matrix, ROC curve) will be displayed or saved as specified in the code.
- The trained model will be saved for future use.

---

## **Prerequisites**
- Python 3.8 or higher.
- A GPU (optional but recommended for faster training).

---

## **Dependencies**
The required libraries are listed in the `requirements.txt` file:
```plaintext
torch==1.13.1
torchvision==0.14.1
numpy==1.23.5
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
```

---

## **License**
This project is open-source and available under the [MIT License](LICENSE).
