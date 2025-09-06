# 🚗 CrackAlert: Real-Time Cracks Detection and Severity Classification System  

## 📌 Overview  
**CrackAlert** is a senior design project developed by a team of engineering students to address the costly and time-consuming process of road maintenance. The system provides an efficient and cost-effective solution for detecting and classifying road cracks in real time, making road inspections more streamlined for government authorities.  

---

## 👥 Team Members  
- Aly Ahmed Sarhan  
- Ali Alloush  
- Yaman Shullar ()
- Ahmad Nayfeh  
- Abdulrahman Alahmadi  

---

## 🎤 Elevator Pitch  
Imagine a world where road safety is never compromised by undetected cracks. **CrackAlert** is a revolutionary system that makes crack detection not only efficient but also remarkably cost-effective. Unlike competing systems, CrackAlert can be easily integrated with various mobile vehicles, providing a flexible and scalable solution for road maintenance.  

---

## ❗ Problem Statement  
Road cracks are a significant issue caused by environmental and civilian factors, requiring regular inspection. Existing inspection methods are often **time-consuming and expensive**.  

- According to Mohamad Wajdi, funds for maintaining pavement networks are decreasing while needs are increasing.  
- The **American Automobile Association** reported that pothole damage cost drivers **$26.5 billion in 2021**.  

👉 CrackAlert provides an **innovative real-time detection system** combined with an **optimized maintenance plan** to reduce these costs.  

---

## 📦 Deliverables  
- ✅ AI-supported crack detection algorithm  
- ✅ Severity classification methodology  
- ✅ User-friendly interface for inspection personnel  
- ✅ Optimized maintenance plan  

---

## 🏗️ System Architecture  

### 1. Detection and Classification System  
Runs on an **edge device** (Jetson Nano) for real-time crack detection and severity classification.  
- **Image Collector**: Intel RealSense 435 camera captures RGB + depth frames. [Link to image](https://github.com/yaman-77/Senior-Design-Project/blob/main/Vids%26Images/Intel%20RealSense%20Camera.jpg)
- **Computer Vision Model**: YOLO model detects defects, outputs type and coordinates. [Link to image](https://github.com/yaman-77/Senior-Design-Project/blob/main/Vids%26Images/YOLODetectionProcess.png)
- **Distance & Depth Estimation**: RGB-D camera + LiDAR line scanner measure crack depth.  
- **GPS Module**: Adafruit GPS captures crack locations. [Link to image](https://github.com/yaman-77/Senior-Design-Project/blob/main/Vids%26Images/AdafruitGPS.png)


---

### 2. Optimized Maintenance Plan System  
Generates efficient maintenance routes based on detected cracks.  

- **Data Inputs**: Crack data, user interface inputs, Google Maps API.  
- **Clustering**: Affinity Propagation groups cracks by road distance.  
- **Optimization Model**: Modified Traveling Salesperson Problem to minimize crew distance and prioritize urgent cracks.  

---

### 3. User Interface (Streamlit Web App)  
A three-page app for inspection personnel.  

- **Information Hub**: Map of detected defects with severity, type, and location.  
- **Maintenance Plan Page**: Generate and view optimized maintenance plans.  

![Image: User interface mockup]  

---

## 🔬 Technical Details  

### Dataset  
- **RDD2022 dataset** with **47,420 images** from six countries.  
- Preprocessing: removed empty samples, converted annotations to YOLO format.  
- Balanced dataset with undersampling by class.  
- Split into **train, validation, and test**.  

![Graph: Images per country]  
![Graph: Class distribution across splits]  

---

### Model Performance  
- Optimization model tested on **100 cracks**.  
- Achieved **17.81% reduction** in objective function vs. standard process.  
- Average optimization runtime: **3.6 minutes**.  

![Table: Optimization results vs. baseline]  

---

### On-Site Experiments & Challenges  
The system was installed on a car for live testing.  

- **GPS Module**: Needed external antenna for accuracy.  
- **Jetson Nano**: Configured in deployment mode; debugging harder but ensured hands-free operation.  

![Video: On-site experiment with car installation]  

---

## 💰 Financials  
- **Budget**: 6000 SAR  
- **Spent**: 5046 SAR  
- **Remaining**: 954 SAR  

---

## 🏆 Achievements  
- Awarded **A+ grade** for the project.  
- **Shortlisted among 12 out of 100+ projects** for startup funding support.  

---

## 🛠️ Tech Stack  
- **Hardware**: Jetson Nano, Intel RealSense 435, LiDAR scanner, Adafruit GPS  
- **Software/Frameworks**: YOLO, Streamlit, Python, Google Maps API  
- **Algorithms**: Affinity Propagation, Modified Traveling Salesperson Model  

---

## 📌 Conclusion  
CrackAlert demonstrates how **AI, edge computing, and optimization models** can transform road inspection and maintenance into a cost-effective, scalable, and highly efficient process.  
