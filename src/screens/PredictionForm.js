
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Form, Container, Modal } from 'react-bootstrap';
import axios from 'axios';
import MedicalData1 from './MedicalData1';
import MedicalData2 from './MedicalData2';
import MedicalData3 from './MedicalData3';
import MedicalData4 from './MedicalData4';
import MedicalData5 from './MedicalData5';
import MedicalData6 from './MedicalData6';
import AdditionalQuestions from './AdditionalQuestions';
import '../styles/PredictionForm.css';
import FormSteps from '../components/FormSteps';

// Define the recommendations object
const recommendations = {
  'High risk of heart attack': [
    'Medical Consultation: Immediately consult with a cardiologist to evaluate the heart\'s condition and consider necessary tests like ECG, stress tests, and possibly coronary angiography.',
    'Lifestyle Changes: Adopt a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins. Reduce intake of salt, sugar, and saturated fats.',
    'Physical Activity: Engage in regular physical activity such as walking, jogging, or swimming. Aim for at least 150 minutes of moderate-intensity aerobic activity per week.',
    'Medication: Adhere strictly to prescribed medications for blood pressure, cholesterol, or any other heart conditions.',
    'Smoking Cessation: If the patient smokes, quitting smoking is crucial for heart health.',
    'Stress Management: Practice stress-reducing techniques like yoga, meditation, or deep breathing exercises.'
  ],
  'Low risk of heart attack': [
    'Regular Check-ups: Continue regular check-ups with a healthcare provider to monitor heart health and manage any existing conditions.',
    'Healthy Diet: Maintain a balanced diet to support cardiovascular health.',
    'Exercise: Stay active with regular exercise to keep the heart strong and healthy.',
    'Avoid Risk Factors: Continue avoiding risk factors such as smoking, excessive alcohol consumption, and high-stress levels.',
    'Monitor Symptoms: Be vigilant about any new or worsening symptoms such as chest pain, shortness of breath, or unusual fatigue, and seek medical advice promptly if they occur.'
  ]
};

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    height: '',
    weight: '',
    age: '',
    thalach: '',
    trtbps_winsorize: '',
    oldpeak_winsorize_sqrt: '',
    sex_1: '',
    cp_1: '',
    cp_2: '',
    cp_3: '',
    exang_1: '',
    slope_1: '',
    slope_2: '',
    ca_1: '',
    ca_2: '',
    ca_3: '',
    ca_4: '',
    thal_2: '',
    thal_3: '',
    exercise: '',
    smoking: '',
    drinking: '',
    familyHistory: ''
  });

  const [currentPage, setCurrentPage] = useState(1);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      let temp = { ...formData };
      for (const [k, v] of Object.entries(temp)) {
        if (Number(v)) {
          continue;
        }
        temp[k] = parseInt(v);
      }
      const response = await axios.post('http://127.0.0.1:5000/predict', [temp]);
      const riskLevel = response.data[0] === 1 ? 'High risk of heart attack' : 'Low risk of heart attack';
      setPrediction(riskLevel);
      setShowModal(true); 
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (currentPage < 7) setCurrentPage(currentPage + 1);
  };

  const handleBack = () => {
    if (currentPage > 1) setCurrentPage(currentPage - 1);
  };

  const handleReset = () => {
    setFormData({
      firstName: '',
      lastName: '',
      height: '',
      weight: '',
      age: '',
      thalach: '',
      trtbps_winsorize: '',
      oldpeak_winsorize_sqrt: '',
      sex_1: '',
      cp_1: '',
      cp_2: '',
      cp_3: '',
      exang_1: '',
      slope_1: '',
      slope_2: '',
      ca_1: '',
      ca_2: '',
      ca_3: '',
      ca_4: '',
      thal_2: '',
      thal_3: '',
      exercise: '',
      smoking: '',
      drinking: '',
      familyHistory: ''
    });
    setCurrentPage(1);
    setPrediction(null);
    setLoading(false);
    setError(null);
  };

  return (
    <Container className="prediction-form">
      <FormSteps
        step1={currentPage >= 1}
        step2={currentPage >= 2}
        step3={currentPage >= 3}
        step4={currentPage >= 4}
        step5={currentPage >= 5}
        step6={currentPage >= 6}
        step7={currentPage >= 7}
      />
      
      <form onSubmit={handleSubmit}>
        {currentPage === 1 && <MedicalData1 formData={formData} handleChange={handleChange} />}
        {currentPage === 2 && <MedicalData2 formData={formData} handleChange={handleChange} />}
        {currentPage === 3 && <MedicalData3 formData={formData} handleChange={handleChange} />}
        {currentPage === 4 && <MedicalData4 formData={formData} handleChange={handleChange} />}
        {currentPage === 5 && <MedicalData5 formData={formData} handleChange={handleChange} />}
        {currentPage === 6 && <MedicalData6 formData={formData} handleChange={handleChange} />}
        {currentPage === 7 && <AdditionalQuestions formData={formData} handleChange={handleChange} />}
        
        <div className="form-navigation">
          <Button className='btnClass' variant="primary" size="lg" onClick={handleBack} disabled={currentPage === 1}>Back</Button>
          {currentPage < 7 ? (
            <Button className='btnClass' variant="primary" size="lg" onClick={handleNext}>Next</Button>
          ) : (
            <>
              <Button type="submit" variant="primary" size="lg" disabled={loading}>Predict</Button>
              <Button type="button" variant="primary" size="lg" onClick={handleReset} disabled={loading}>Start New Search</Button>
            </>
          )}
        </div>
      </form>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      
      {/* Modal for showing prediction results */}
      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Prediction Results</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {prediction !== null && (
            <div className="prediction">
              <p><span>Prediction:  </span>
              <span><h5 style={{ display: 'inline' }}>{prediction}</h5> </span>
              </p>
              <h2>Recommendations: </h2>
              <ul>
                {recommendations[prediction].map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
              <div className="recommendation-images">
                <div className="image-col"><img src={require('../images/cardiologue.jpg')} alt="Medical Consultation" /></div>
                <div className="image-col"><img src={require('../images/fruits-vegetables.webp')} alt="Healthy Diet" /></div>
                <div className="image-col"><img src={require('../images/physical_activity.jpeg')} alt="Physical Activity" /></div>
                <div className="image-col"><img src={require('../images/pharmacist-img.jpg')} alt="Medication" /></div>
                <div className="image-col"><img src={require('../images/stop_smoking.jpg')} alt="No Smoking" /></div>
                <div className="image-col"><img src={require('../images/yoga-img.jpeg')} alt="Stress Management" /></div>
              </div>
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModal(false)}>Close</Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default PredictionForm;
