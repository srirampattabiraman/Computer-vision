CODING DRILL DOWN

Objective:

To maintain consistent accuracy in a range of 99.4% within 15 epochs.

Dataset:
MNIST

HOW?:

With the help of batch normalization , Global average pooling, and other parameters such as LR schedular, Image augumentation etc.

STEPS: 5

Observations:

Having lower number of kernels in the convolution block gave an accuracy < 99.4 %.
Dropouts needs to be negligible else accuracy is dropping out :)
The validation accuracy was always less than the training accuracy.

CODE BLOCKS EXPLANATION:

STEP 1 EVA4_S5_Step1_BN&GAP

Introduced Batch Normalization and Global average pooling with the existing architecture.

Played with different values of dropout and observed that dropout needs to be less than 0.5 else training accuracy is very less.

GAP should not be the last layer, it should be before last convolution inorder to replace Fully connected layer

Having a dobut that if we add batch-Norm after relu will it impact the performace in terms of accuracy?

Total Parameters: 9,840
Max Accuracy achieved within 15 epochs : 99.2%

STEP 2 EVA4_S5_Step2_IA

INTRODUCING Regularization strategy called Image Augumentation

transforms.RandomRotation((-7.0, 7.0), fill=(1,))

++++++

EPOCH: 10
Loss=0.012849663384258747 Batch_id=468 Accuracy=98.48: 100%|██████████| 469/469 [00:12<00:00, 37.71it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0233, Accuracy: 9919/10000 (99.19%)

EPOCH: 11
Loss=0.08857490867376328 Batch_id=468 Accuracy=98.52: 100%|██████████| 469/469 [00:12<00:00, 36.93it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.40%)

++++

Could witness 99.4 at epoch 11 but that was not consistent enough.

STEP 3 EVA4_S5_Step3_Schedular

Added Learning Rate schedular along with Image augumentation.

Epoch: 12 LR: 0.0010000000000000002
Loss=0.023297637701034546 Batch_id=468 Accuracy=99.15: 100%|██████████| 469/469 [00:15<00:00, 35.25it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0185, Accuracy: 9943/10000 (99.43%)

Epoch: 13 LR: 0.0010000000000000002
Loss=0.0064370580948889256 Batch_id=468 Accuracy=99.12: 100%|██████████| 469/469 [00:15<00:00, 36.71it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0184, Accuracy: 9942/10000 (99.42%)

Epoch: 14 LR: 0.0010000000000000002
Loss=0.007203350309282541 Batch_id=468 Accuracy=99.12: 100%|██████████| 469/469 [00:15<00:00, 30.38it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0188, Accuracy: 9945/10000 (99.45%)

Epoch: 15 LR: 0.0010000000000000002
Loss=0.01633496582508087 Batch_id=468 Accuracy=99.15: 100%|██████████| 469/469 [00:15<00:00, 31.19it/s]
Test set: Average loss: 0.0183, Accuracy: 9941/10000 (99.41%)

The result was consistent from epoch 12.



