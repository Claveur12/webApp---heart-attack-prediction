// import React from 'react';

// const MedicalData5 = ({ formData, handleChange }) => (
//   <div>
//     <h2>Major Vessels and Thalassemia</h2>
//     <div className="form-row">
//       <div className="form-group">
//         <label>Number of Major Vessels (1)</label>
//         <input type="number" name="ca_1" value={formData.ca_1} onChange={handleChange} required />
//       </div>
//       <div className="form-group">
//         <label>Number of Major Vessels (2)</label>
//         <input type="number" name="ca_2" value={formData.ca_2} onChange={handleChange} required />
//       </div>
//     </div>
//     <div className="form-row">
//       <div className="form-group">
//         <label>Number of Major Vessels (3)</label>
//         <input type="number" name="ca_3" value={formData.ca_3} onChange={handleChange} required />
//       </div>
//       <div className="form-group">
//         <label>Number of Major Vessels (4)</label>
//         <input type="number" name="ca_4" value={formData.ca_4} onChange={handleChange} required />
//       </div>
//     </div>
//     <div className="form-row">
//       <div className="form-group">
//         <label>Thalassemia (2 = normal)</label>
//         <input type="number" name="thal_2" value={formData.thal_2} onChange={handleChange} required />
//       </div>
//       <div className="form-group">
//         <label>Thalassemia (3 = reversible defect)</label>
//         <input type="number" name="thal_3" value={formData.thal_3} onChange={handleChange} required />
//       </div>
//     </div>
//   </div>
// );

// export default MedicalData5;
import React from 'react';
import { Container, Col, Row, Form, Image } from 'react-bootstrap';
import heartAttackOlderPeople from '../images/heart-attack-older-people.jpg';
import '../styles/PredictionForm.css';

const MedicalData5 = ({ formData, handleChange }) => (
  <Container className="mt-4 medical-data">
    <h2>Number of Major Vessels</h2>
    <Row>
      {/* Left Column: Form Inputs */}
      <Col md={6} className='column-field-data'>
        <Form className='form-group-data'>
          <Form.Group controlId="formCa1" className='form-group'>
            <Form.Label>Number of Major Vessels (1)</Form.Label>
            <Form.Control
              type="number"
              name="ca_1"
              value={formData.ca_1}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formCa2" className='form-group'>
            <Form.Label>Number of Major Vessels (2)</Form.Label>
            <Form.Control
              type="number"
              name="ca_2"
              value={formData.ca_2}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formCa3" className='form-group'>
            <Form.Label>Number of Major Vessels (3)</Form.Label>
            <Form.Control
              type="number"
              name="ca_3"
              value={formData.ca_3}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formCa4" className='form-group'>
            <Form.Label>Number of Major Vessels (4)</Form.Label>
            <Form.Control
              type="number"
              name="ca_4"
              value={formData.ca_4}
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
              src={heartAttackOlderPeople}
              alt="Personal Data"
              fluid
            className="d-block w-100" 
            />
          </Col>       
        </Row>
        <Row className='heart-attack-comment'>
          <Col>
          <p>
          <h5>Who is at Risk for a Heart Attack?</h5><b/>
          <span className='heart-text-comment'>Risk factors include high blood pressure, 
            high cholesterol, smoking, obesity, diabetes, inactivity, and family history.</span>
          </p>          
          </Col>
        </Row>
      </Col>
    </Row>
  </Container>
);

export default MedicalData5;
