// import React from 'react';

// const MedicalData4 = ({ formData, handleChange }) => (
//   <div>
//     <h2>Exercise and ST Segment</h2>
//     <div className="form-row">
//       <div className="form-group">
//         <label>Exercise Induced Angina (1 = yes, 0 = no)</label>
//         <input type="number" name="exang_1" value={formData.exang_1} onChange={handleChange} required />
//       </div>
//       <div className="form-group">
//         <label>Slope of Peak Exercise ST Segment (1 = flat)</label>
//         <input type="number" name="slope_1" value={formData.slope_1} onChange={handleChange} required />
//       </div>
//     </div>
//     <div className="form-row">
//       <div className="form-group">
//         <label>Slope of Peak Exercise ST Segment (2 = upsloping)</label>
//         <input type="number" name="slope_2" value={formData.slope_2} onChange={handleChange} required />
//       </div>
//     </div>
//   </div>
// );

// export default MedicalData4;


import React from 'react';
import { Container, Col, Row, Form, Image } from 'react-bootstrap';
import heartAttackSymptoms from '../images/heart-attack-symptoms.jpg';
import '../styles/PredictionForm.css';

const MedicalData4 = ({ formData, handleChange }) => (
  <Container className="mt-4 medical-data">
    <h2>Exercise and ST Segment</h2>
    <Row>
      {/* Left Column: Form Inputs */}
      <Col md={6} className='column-field-data'>
        <Form className='form-group-data'>
          <Form.Group controlId="formExang" className='form-group'>
            <Form.Label>Exercise Induced Angina (1 = yes, 0 = no)</Form.Label>
            <Form.Control
              type="number"
              name="exang_1"
              value={formData.exang_1}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formSlope1" className='form-group'>
            <Form.Label>Slope of Peak Exercise ST Segment (1 = flat)</Form.Label>
            <Form.Control
              type="number"
              name="slope_1"
              value={formData.slope_1}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formSlope2" className='form-group'>
            <Form.Label>Slope of Peak Exercise ST Segment (2 = upsloping)</Form.Label>
            <Form.Control
              type="number"
              name="slope_2"
              value={formData.slope_2}
              onChange={handleChange}
              required
            />
          </Form.Group>
        </Form>
      </Col>

      {/* Right Column: Image */}
      <Col md={6} className='column-field-data'>
        <Row>
          <Col>
            <Image
              src={heartAttackSymptoms}
              alt="Personal Data"
              fluid
            className="d-block w-100" 
            />
          </Col>       
        </Row>
        <Row className='heart-attack-comment'>
          <Col>
          <p>
          <h5>What are the Symptoms of a Heart Attack?</h5><b/>
          <span className='heart-text-comment'>Symptoms include chest pain, shortness of breath, 
            nausea, and pain spreading to shoulders, arms, neck, or jaw.</span>
          </p>          
          </Col>
        </Row>
      </Col>
    </Row>
  </Container>
);

export default MedicalData4;
