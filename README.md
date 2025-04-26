# Msc_Data_Science_Project

**Socioeconomic Status (SES) Prediction Web Application**

*Project Overview*
This project develops an end-to-end machine learning application that predicts an individual’s socioeconomic status (SES) using demographic and employment attributes. The system classifies whether an individual earns above or below $50,000 annually using machine learning models.

The final deployed model is Gradient Boosting, chosen for its high accuracy (80%) and AUC-ROC (0.87), ensuring strong real-world predictive performance.


Project Structure
app.py — Flask application code.

final_gradient_boosting_model.pkl.gz — Compressed trained model (saved in the main project folder).

index.html and result.html — HTML templates for the web interface (inside /templates/ folder).

README.md — Project overview and instructions.

requirements.txt — Python dependencies list.

*How to Run Locally*
1. Clone the Repository
```python
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
2. Install Required Packages
```
pip install -r requirements.txt
```
3. Run the Flask Application
```
python app.py
```
4. Access the Web App
Open your browser and navigate to:
http://127.0.0.1:81

*Features*

Predict SES using five key demographic/employment attributes:

Occupation (OCCP)

Work Hours per Week (WKHP)

Education Level (SCHL)

Age (AGEP)

Sex (SEX)

Clean, Bootstrap-styled user interface (online Bootstrap CDN used).

Real-time predictions without saving any user data.

Fully interactive experience from input to prediction.


Live Testing URL
👉 Test the deployed application here (for demonstration purposes only):
[Insert Your Replit App URL Here]

Note:
This deployment is for academic demonstration only and is ethically compliant (no real user data is collected or stored).

Key Highlights
Dataset: Subset of the Folktables dataset derived from the American Community Survey (ACS).

Final Model: Gradient Boosting Classifier.

Performance Metrics:

Accuracy: 80%

AUC-ROC: 0.87

Deployment Platform: Replit.

Future Improvements
Expand the model to cover more detailed SES attributes.

Improve feature engineering for even stronger prediction accuracy.

Upgrade hosting to a production environment (e.g., AWS, Heroku) for scalable use.


*License*
This project is licensed under the MIT License.




