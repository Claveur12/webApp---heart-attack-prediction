// import React from 'react';

// const MedicalData6 = ({ formData, handleChange }) => (
//   <div>
//     <h2>Thalassemia</h2>
//     <div className="form-row">
//       <div className="form-group">
//         <label>Thalassemia (2 = normal)</label>
//         <input
//           type="number"
//           name="thal_2"
//           value={formData.thal_2}
//           onChange={handleChange}
//           required
//         />
//       </div>
//       <div className="form-group">
//         <label>Thalassemia (3 = reversible defect)</label>
//         <input
//           type="number"
//           name="thal_3"
//           value={formData.thal_3}
//           onChange={handleChange}
//           required
//         />
//       </div>
//     </div>
//   </div>
// );

// export default MedicalData6;

import React from 'react';
import { Container, Col, Row, Form, Image } from 'react-bootstrap';
import heartAttackAvoid from '../images/heart-attack-avoid.jpg';
import '../styles/PredictionForm.css';

const MedicalData6 = ({ formData, handleChange }) => (
  <Container className="mt-4 medical-data">
    <h2>Thalassemia</h2>
    <Row>
      {/* Left Column: Form Inputs */}
      <Col md={6} className='column-field-data'>
        <Form className='form-group-data'>
          <Form.Group controlId="formThal2" className='form-group'>
            <Form.Label>Thalassemia (2 = normal)</Form.Label>
            <Form.Control
              type="number"
              name="thal_2"
              value={formData.thal_2}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formThal3" className='form-group'>
            <Form.Label>Thalassemia (3 = reversible defect)</Form.Label>
            <Form.Control
              type="number"
              name="thal_3"
              value={formData.thal_3}
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
              src={heartAttackAvoid}
              alt="Personal Data"
              fluid
            className="d-block w-100" 
            />
          </Col>       
        </Row>
        <Row className='heart-attack-comment'>
          <Col>
          <p>
          <h5>How Can Heart Attacks be Prevented? </h5><b/>
          <span className='heart-text-comment'>Prevent heart attacks by exercising,
             eating a balanced diet, avoiding smoking, managing stress, and regular check-ups.</span>
          </p>          
          </Col>
        </Row>
      </Col>
    </Row>
  </Container>);

export default MedicalData6;
