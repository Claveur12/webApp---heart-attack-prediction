
import React from 'react';
import { Container, Col, Row, Form, Image } from 'react-bootstrap';
import heartAttackImg from '../images/heart-attack-ambulance.jpeg';
import '../styles/PredictionForm.css';

const AdditionalQuestions = ({ formData, handleChange }) => (
  <Container className="mt-4 additional-questions">
    <h2>Additional Questions</h2>
    <Row>
      {/* Left Column: Form Inputs */}
      <Col md={6} className='column-field-data'>
        <Form className='form-group-data'>
          <Form.Group controlId="formExercise" className='form-group'>
            <Form.Label>Exercise</Form.Label>
            <Form.Control
              type="number"
              name="exercise"
              value={formData.exercise}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formSmoking" className='form-group'>
            <Form.Label>Smoking</Form.Label>
            <Form.Control
              type="number"
              name="smoking"
              value={formData.smoking}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formDrinking">
            <Form.Label>Drinking</Form.Label>
            <Form.Control
              type="number"
              name="drinking"
              value={formData.drinking}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formFamilyHistory">
            <Form.Label>Family History of Heart Attack</Form.Label>
            <Form.Control
              type="number"
              name="familyHistory"
              value={formData.familyHistory}
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
              src={heartAttackImg}
              alt="Personal Data"
              fluid
            className="d-block w-100" 
            />
          </Col>       
        </Row>
        <Row className='heart-attack-comment'>
          <Col>
          <p>
          <h5>What Should You Do if You Suspect a Heart Attack?</h5><b/>
          <span className='heart-text-comment'>Call emergency services, stay calm, rest, and chew an aspirin if available.</span>
          </p>          
          </Col>
        </Row>
      </Col>
    </Row>
  </Container>
);

export default AdditionalQuestions;
