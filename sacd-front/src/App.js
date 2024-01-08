import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const AppContainer = styled.div`
  font-family: Arial, sans-serif;
  text-align: center;
  color: #333;
`;

const FormContainer = styled.div`
  background-color: #3498db;
  padding: 20px;
  border-radius: 10px;
  margin: 20px;
`;

const FormField = styled.div`
  display: flex;
  flex-direction: column;
  margin-bottom: 15px;

  label {
    margin-bottom: 5px;
    color: #fff;
  }

  input {
    padding: 8px;
    font-size: 16px;
  }
`;

const SubmitButton = styled.button`
  background-color: #2ecc71;
  color: #fff;
  padding: 10px 20px;
  font-size: 16px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
`;

const initialFormData = {
  min_salary: 1000,
  remote_allowed: true,
  formatted_experience_level: 'Entry-Level',
  location: 'United States',
  employee_count: 57,
  company: 'Amazon',
};

const App = () => {
  const [formData, setFormData] = useState(initialFormData);
  const [submitted, setSubmitted] = useState(false);
  const [recommendation, setRecommendation] = useState("");

  const handleInputChange = (field, value) => {
    setFormData({
      ...formData,
      [field]: value,
    });
  };

  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/sendData', formData);
      console.log('Server Response:', response.data);
      setSubmitted(true);
      setRecommendation(response.data);
    } catch (error) {
      console.error('Error submitting data:', error);
    }
    setSubmitted(true);
  };

  return (
    <AppContainer>
        <FormContainer>
          <h2>Job Information Form</h2>
          <p>Fill in the details:</p>
          <form>
            <FormField>
              <label>Minimum Salary:</label>
              <input
                type="number"
                value={formData.min_salary}
                onChange={(e) => handleInputChange('min_salary', e.target.value)}
              />
            </FormField>
            <FormField>
              <label>Remote Allowed:</label>
              <input
                type="checkbox"
                checked={formData.remote_allowed}
                onChange={(e) => handleInputChange('remote_allowed', e.target.checked)}
              />
            </FormField>
            <FormField>
              <label>Experience Level:</label>
              <input
                type="text"
                value={formData.formatted_experience_level}
                onChange={(e) =>
                  handleInputChange('formatted_experience_level', e.target.value)
                }
              />
            </FormField>
            <FormField>
              <label>Location:</label>
              <input
                type="text"
                value={formData.location}
                onChange={(e) => handleInputChange('location', e.target.value)}
              />
            </FormField>
            <FormField>
              <label>Employee Count:</label>
              <input
                type="number"
                value={formData.employee_count}
                onChange={(e) => handleInputChange('employee_count', e.target.value)}
              />
            </FormField>
            <FormField>
              <label>Company:</label>
              <input
                type="text"
                value={formData.company}
                onChange={(e) => handleInputChange('company', e.target.value)}
              />
            </FormField>
            <SubmitButton type="button" onClick={handleSubmit}>
              Submit
            </SubmitButton>
          </form>
        </FormContainer>
        {submitted && (
        <FormContainer>
          <div>
            <h1 style={{ color: 'white' }}>Your industry recommendation is: {recommendation}</h1>
          </div>
        </FormContainer>
        )}
    </AppContainer>
  );
};

export default App;
