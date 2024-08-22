// import React from 'react';

// const MedicalData3 = ({ formData, handleChange }) => (
//   <div>
//     <h2>Chest Pain Types</h2>
//     <div className="form-row">
//       <div className="form-group">
//         <label>Chest Pain Type (1 = typical angina)</label>
//         <input type="number" name="cp_1" value={formData.cp_1} onChange={handleChange} required />
//       </div>
//       <div className="form-group">
//         <label>Chest Pain Type (2 = atypical angina)</label>
//         <input type="number" name="cp_2" value={formData.cp_2} onChange={handleChange} required />
//       </div>
//       <div className="form-group">
//         <label>Chest Pain Type (3 = non-anginal pain)</label>
//         <input type="number" name="cp_3" value={formData.cp_3} onChange={handleChange} required />
//       </div>
//     </div>
//   </div>
// );

// export default MedicalData3;

import React from 'react';
import { Container, Col, Row, Form, Image } from 'react-bootstrap';
import heartAttackOccurence from '../images/heart-failure.webp';
import '../styles/PredictionForm.css';

const MedicalData3 = ({ formData, handleChange }) => (
  <Container className="mt-4 medical-data">
    <h2>Chest Pain Types</h2>
    <Row className='column-field-data'>
      {/* Left Column: Form Inputs */}
      <Col md={6} className='column-field-data'>
        <Form className='form-group-data'>
          <Form.Group controlId="formCp1" className='form-group'>
            <Form.Label>Chest Pain Type (1 = typical angina)</Form.Label>
            <Form.Control
              type="number"
              name="cp_1"
              value={formData.cp_1}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formCp2" className='form-group'>
            <Form.Label>Chest Pain Type (2 = atypical angina)</Form.Label>
            <Form.Control
              type="number"
              name="cp_2"
              value={formData.cp_2}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formCp3" className='form-group'>
            <Form.Label>Chest Pain Type (3 = non-anginal pain)</Form.Label>
            <Form.Control
              type="number"
              name="cp_3"
              value={formData.cp_3}
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
              src={heartAttackOccurence}
              alt="Personal Data"
              fluid
            className="d-block w-100" 
            />
          </Col>       
        </Row>
        <Row className='heart-attack-comment'>
          <Col>
          <p>
          <h5>How Does a Heart Attack Occur?</h5> <b/><span className='heart-text-comment'>A heart attack happens when 
            a part of the heart muscle doesn't get enough blood due to a blocked artery.</span>
          </p>          
          </Col>
        </Row>
      </Col>
    </Row>
  </Container>
);

export default MedicalData3;
