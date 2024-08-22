
import React from 'react';
import { Container, Col, Row, Form, Image } from 'react-bootstrap';
import heartAttackCauses from '../images/heart-attack-causes.jpg';
import '../styles/PredictionForm.css';

const MedicalData2 = ({ formData, handleChange }) => (
  <Container className="mt-4 medical-data">
    <h2>Medical Data</h2>
    <Row className='column-field-data'>
      {/* Left Column: Form Inputs */}
      <Col md={6} className='column-field-data'>
        <Form className='form-group-data'>
          <Form.Group controlId="formAge" className='form-group'>
            <Form.Label>Age</Form.Label>
            <Form.Control
              type="number"
              name="age"
              value={formData.age}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formThalach" className='form-group'>
            <Form.Label>Maximum Heart Rate Achieved (thalach)</Form.Label>
            <Form.Control
              type="number"
              name="thalach"
              value={formData.thalach}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formTrtbpsWinsorize" className='form-group'>
            <Form.Label>Resting Blood Pressure (trtbps_winsorize)</Form.Label>
            <Form.Control
              type="number"
              name="trtbps_winsorize"
              value={formData.trtbps_winsorize}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formOldpeakWinsorizeSqrt" className='form-group'>
            <Form.Label>ST Depression Induced by Exercise (oldpeak_winsorize_sqrt)</Form.Label>
            <Form.Control
              type="number"
              name="oldpeak_winsorize_sqrt"
              value={formData.oldpeak_winsorize_sqrt}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formSex" className='form-group'>
            <Form.Label>Sex (1 = male, 0 = female)</Form.Label>
            <Form.Control
              type="text"
              name="sex"
              value={formData.sex}
              onChange={handleChange}
              required
            />
          </Form.Group>
        </Form>
      </Col>

      {/* Right Column: Image */}
      <Col md={6} className='column-field-data'>
        <Row >
          <Col>
            <Image
              src={heartAttackCauses}
              alt="Personal Data"
              fluid
            className="d-block w-100" 
            />
          </Col>       
        </Row>
        <Row className='heart-attack-comment'>
          <Col>
          <p>
          <h5>What Causes a Heart Attack?</h5><b/> <span className='heart-text-comment'>Heart attacks are mainly caused by coronary 
            heart disease, where plaque builds up in the heart's arteries.</span>
          </p>          
          </Col>
        </Row>
      </Col>
    </Row>
  </Container>
);

export default MedicalData2;
