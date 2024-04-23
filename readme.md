## CoMN V1.0

CoMN platform was developed by **Prof. Peng Huang's research group(Peking University)**. The platfrom is proposed for designers to fast verify and further optimize the designments of non-volatile memory based neural network accelerators. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the **MIT License**.

### Get Started

CoMN is composed of three main parts: **algorithm adapter**, which aims at improving inference accuracy of neural networks by considering the nonidealities of NVM devices; **mapper**, which is developed to automatically map CNN models to CIM chips through optimizing pipeline, weight transformation, partition, and placement; and **hardware optimizer**, which is developed to search hardware microarchitecture and circuit design space in the early design stage.

Before running the algorithm adapter/mapper/hardware optimizer, first compile the C++ files to executable file:
```bash
cd src/refactor
chmod +x run.sh
./run.sh
# Go back to CoMN base directory Before executing the following command
cd cacti-master
make
# Go back to CoMN base directory Before executing the following command
cd ORION3_0
make
```

Then, use the following instructions to run algorithm adapter/mapper/hardware optimizer:
```bash
cd src
python Accuracy_optimizer.py # adapter
python Mapping_optimizer.py # mapper
python Performance_optimizer.py # hardware optimizer
```
You can change configurations of CoMN in directory "Parameters", where all the json files except epoch.json can be reconfigured, and all the txt files are just intermediate products.
The generations of the python files are located in "generate_data/tcad".

### GUI for CoMN
We have also developed a Web GUI for CoMN. You can access the platform via http://101.42.97.22:8081/index.html. You need to register first before using platform.
Still, we strongly recommend you download the source code and use CoMN platform locally due to limited GPU resources on our server.

### Contact Us
Lixia Han: lixiahan@pku.edu.cn

Peng Huang: phwang@pku.edu.cn

Siyuan Chen (Now maintaining the code): 3197571813@mail.nwpu.edu.cn

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:

**Lixia, Han & Pan, Renjie & Zhou, Zheng & Lu, Hairuo & Chen, Yiyang & Haozhang, Yang & Huang, Peng & Sun, Guangyu & Kang, Jinfeng. (2024).** **CoMN: Algorithm-Hardware Co-Design Platform for Non-Volatile Memory Based Convolutional Neural Network Accelerators. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems. PP. 1-1. 10.1109/TCAD.2024.3358220.** 
