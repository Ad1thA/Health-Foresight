import express from 'express';
import * as dotenv from 'dotenv';
import cors from 'cors';
import { spawn } from 'child_process'; // Import child_process




dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
    res.send('Hello World');
});

// New route to handle form submission and run the Python script
app.post('/api/diabetes', (req, res) => {
    const { no_preg, glucose, bl_pres, skin_thick, insulin, bmi, dia_func, age } = req.body;
    // res.send(req.body)
   
    
    
    // const python = spawn('python', ['./diabetes_prediction.py', 4, 199, 70, 1, 4, 55.8, 0.553, 65 ]);
    const python = spawn('python', ['./diabetes_prediction.py', no_preg, glucose, bl_pres, skin_thick, insulin, bmi, dia_func, age]);

    let dataToSend = '';
    python.stdout.on('data', function (data) {
        dataToSend += data.toString();
        console.log(dataToSend);
    });

    python.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    python.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        
        // res.json(JSON.parse(dataToSend));
        
        res.setHeader('Content-Type', 'application/json');
        
        res.send(dataToSend);
    });
    
});

app.post('/api/heartdesease', (req, res) => {
    const { age, sex, chestPain, restingBloodPressure, serumCholesterol,fastingBloodSugar, restingECG, maxHeartRate,exerciseAngina,stDepression,stSlope,majorVessels, thal} = req.body;
    // res.send(req.body)
   
    
    
    const python = spawn('python', ['./heart_disease_prediction.py', age, sex, chestPain, restingBloodPressure, serumCholesterol, fastingBloodSugar, restingECG, maxHeartRate,exerciseAngina,stDepression,stSlope,majorVessels, thal]);
    // const python = spawn('python', ['./diabetes_prediction.py', no_preg, glucose, bl_pres, skin_thick, insulin, bmi, dia_func, age]);

    let dataToSend = '';
    python.stdout.on('data', function (data) {
        dataToSend += data.toString();
        
    });

    python.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    python.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        
        // res.json(JSON.parse(dataToSend));
        
        res.setHeader('Content-Type', 'application/json');
        
        res.send(dataToSend);
    });
    
});




const startServer = async () => {
    try {
        
        app.listen(process.env.PORT, () => {
            console.log(`Server is running on port http://localhost:${process.env.PORT}`);
        });
    } catch (error) {
        console.log(error);
    }
}

startServer();
