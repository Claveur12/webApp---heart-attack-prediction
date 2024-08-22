
// import React from 'react';
// import { useLocation } from 'react-router-dom';
// import fruitVegetableImage from '../images/fruits-vegetables.webp';
// import physicalActivity from '../images/physical_activity.jpeg';
// import medication from '../images/pharmacist-img.jpg';
// import noSmoking from '../images/stop_smoking.jpg';
// import stressManagement from '../images/yoga-img.jpeg';
// import '../styles/PredictionForm.css';

// const recommendations = {
//   1: [
//     'Medical Consultation: Immediately consult with a cardiologist to evaluate the heart\'s condition and consider necessary tests like ECG, stress tests, and possibly coronary angiography.',
//     'Lifestyle Changes: Adopt a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins. Reduce intake of salt, sugar, and saturated fats.',
//     'Physical Activity: Engage in regular physical activity such as walking, jogging, or swimming. Aim for at least 150 minutes of moderate-intensity aerobic activity per week.',
//     'Medication: Adhere strictly to prescribed medications for blood pressure, cholesterol, or any other heart conditions.',
//     'Smoking Cessation: If the patient smokes, quitting smoking is crucial for heart health.',
//     'Stress Management: Practice stress-reducing techniques like yoga, meditation, or deep breathing exercises.'
//   ],
//   0: [
//     'Regular Check-ups: Continue regular check-ups with a healthcare provider to monitor heart health and manage any existing conditions.',
//     'Healthy Diet: Maintain a balanced diet to support cardiovascular health.',
//     'Exercise: Stay active with regular exercise to keep the heart strong and healthy.',
//     'Avoid Risk Factors: Continue avoiding risk factors such as smoking, excessive alcohol consumption, and high-stress levels.',
//     'Monitor Symptoms: Be vigilant about any new or worsening symptoms such as chest pain, shortness of breath, or unusual fatigue, and seek medical advice promptly if they occur.'
//   ]
// };

// const ResultPrediction = () => {
//   const location = useLocation();
//   const { prediction } = location.state;

//   return (
//     <div className="result-prediction ">
//       <h2>Prediction: {prediction}</h2>
//       <h3>Recommendations:</h3>
//       <ul>
//         {recommendations[prediction]?.map((rec, index) => (
//           <li key={index}>{rec}</li>
//         ))}
//       </ul>
//       <div className="recommendation-images">
//         <img src={fruitVegetableImage} alt="Eat fruits and vegetables" />
//         <img src={physicalActivity} alt="Engage in physical activity" />
//         <img src={medication} alt="Take prescribed medications" />
//         <img src={noSmoking} alt="Stop smoking" />
//         <img src={stressManagement} alt="Practice stress management" />
//       </div>
//     </div>
//   );
// };

// export default ResultPrediction;
