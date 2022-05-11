# Semi-supervised Pytorch Medical Segmentation


## Recent Updates
* 2022.5.11 The SSL codes are released.


## Requirements
* PyTorch > 1.7
* torchio <= 0.18.20
* python >= 3.7

## Notice
* You can modify **hparam.py** to determine whether 2D or 3D segmentation.
* We provide some SSL algorithms on 2D and 3D segmentation.
* The project is highly based on [Pytorch-Medical-Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation). 




## Done
* SSL
- [x] Mean-Teacher
- [x] Cross-Pesudo-Supervision
- [x] Entropy-Minimization
- [x] Interpolation-Consistency
- [x] Co-Training
- [x] Cross-Teaching
- [x] Uncertain-Aware-Mean-Teacher
- [x] Pi-Model

## By The Way
This project is not perfect and there are still many problems. If you are using this project and would like to give the author some feedbacks, you can send [Me](elliszkn@163.com) an email.

## Acknowledgements
This repository is an unoffical PyTorch implementation of Semi-Supervised Learning Medical segmentation in 3D and 2D and highly based on [Pytorch-Medical-Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thank you for the above repo. The project is done with the supervisions of [Prof. Ruoxiu Xiao](http://enscce.ustb.edu.cn/Teach/TeacherList/2020-10-16/114.html) and [Dr. Cheng Chen](b20170310@xs.ustb.edu.cn).
