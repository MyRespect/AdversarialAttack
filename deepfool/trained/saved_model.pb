��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.0-beta12v2.0.0-beta0-16-g1d91213fe78��
~
conv2d/kernelVarHandleOp*
shape: *
shared_nameconv2d/kernel*
dtype0*
_output_shapes
: 
�
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
: 
n
conv2d/biasVarHandleOp*
shape: *
shared_nameconv2d/bias*
dtype0*
_output_shapes
: 
�
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
: 
�
conv2d_1/kernelVarHandleOp*
shape: @* 
shared_nameconv2d_1/kernel*
dtype0*
_output_shapes
: 
�
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
: @
r
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: *
shape:@
�
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
u
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	�H@*
shared_namedense/kernel
�
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	�H@*
_class
loc:@dense/kernel
l

dense/biasVarHandleOp*
shape:@*
shared_name
dense/bias*
dtype0*
_output_shapes
: 
�
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:@
x
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@ *
shared_namedense_1/kernel
�
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@ 
p
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
dtype0*
_output_shapes
: *
shape: 
�
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_1/bias
x
dense_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
: 
*
shared_namedense_2/kernel
�
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes

: 

p
dense_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*
shared_namedense_2/bias
�
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
:

f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
}
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: *
_class
loc:@Adam/iter
j
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: 
�
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
�
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name
Adam/decay
�
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/learning_rate
�
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*%
_class
loc:@Adam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 
�
Adam/conv2d/kernel/mVarHandleOp*
shape: *%
shared_nameAdam/conv2d/kernel/m*
dtype0*
_output_shapes
: 
�
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*'
_class
loc:@Adam/conv2d/kernel/m*
dtype0*&
_output_shapes
: 
|
Adam/conv2d/bias/mVarHandleOp*
shape: *#
shared_nameAdam/conv2d/bias/m*
dtype0*
_output_shapes
: 
�
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
dtype0*
_output_shapes
: *%
_class
loc:@Adam/conv2d/bias/m
�
Adam/conv2d_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*'
shared_nameAdam/conv2d_1/kernel/m
�
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*
dtype0*&
_output_shapes
: @*)
_class
loc:@Adam/conv2d_1/kernel/m
�
Adam/conv2d_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*%
shared_nameAdam/conv2d_1/bias/m
�
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*'
_class
loc:@Adam/conv2d_1/bias/m*
dtype0*
_output_shapes
:@
�
Adam/dense/kernel/mVarHandleOp*
shape:	�H@*$
shared_nameAdam/dense/kernel/m*
dtype0*
_output_shapes
: 
�
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*&
_class
loc:@Adam/dense/kernel/m*
dtype0*
_output_shapes
:	�H@
z
Adam/dense/bias/mVarHandleOp*"
shared_nameAdam/dense/bias/m*
dtype0*
_output_shapes
: *
shape:@
�
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*
_output_shapes
:@*$
_class
loc:@Adam/dense/bias/m
�
Adam/dense_1/kernel/mVarHandleOp*
shape
:@ *&
shared_nameAdam/dense_1/kernel/m*
dtype0*
_output_shapes
: 
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*(
_class
loc:@Adam/dense_1/kernel/m*
dtype0*
_output_shapes

:@ 
~
Adam/dense_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *$
shared_nameAdam/dense_1/bias/m
�
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*&
_class
loc:@Adam/dense_1/bias/m*
dtype0*
_output_shapes
: 
�
Adam/dense_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
: 
*&
shared_nameAdam/dense_2/kernel/m
�
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
dtype0*
_output_shapes

: 
*(
_class
loc:@Adam/dense_2/kernel/m
~
Adam/dense_2/bias/mVarHandleOp*$
shared_nameAdam/dense_2/bias/m*
dtype0*
_output_shapes
: *
shape:

�
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*&
_class
loc:@Adam/dense_2/bias/m*
dtype0*
_output_shapes
:

�
Adam/conv2d/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nameAdam/conv2d/kernel/v
�
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*'
_class
loc:@Adam/conv2d/kernel/v*
dtype0*&
_output_shapes
: 
|
Adam/conv2d/bias/vVarHandleOp*#
shared_nameAdam/conv2d/bias/v*
dtype0*
_output_shapes
: *
shape: 
�
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
dtype0*
_output_shapes
: *%
_class
loc:@Adam/conv2d/bias/v
�
Adam/conv2d_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*'
shared_nameAdam/conv2d_1/kernel/v
�
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*)
_class
loc:@Adam/conv2d_1/kernel/v*
dtype0*&
_output_shapes
: @
�
Adam/conv2d_1/bias/vVarHandleOp*
shape:@*%
shared_nameAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
: 
�
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
:@*'
_class
loc:@Adam/conv2d_1/bias/v
�
Adam/dense/kernel/vVarHandleOp*$
shared_nameAdam/dense/kernel/v*
dtype0*
_output_shapes
: *
shape:	�H@
�
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*&
_class
loc:@Adam/dense/kernel/v*
dtype0*
_output_shapes
:	�H@
z
Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*"
shared_nameAdam/dense/bias/v
�
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
dtype0*
_output_shapes
:@*$
_class
loc:@Adam/dense/bias/v
�
Adam/dense_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@ *&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*(
_class
loc:@Adam/dense_1/kernel/v*
dtype0*
_output_shapes

:@ 
~
Adam/dense_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *$
shared_nameAdam/dense_1/bias/v
�
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*&
_class
loc:@Adam/dense_1/bias/v*
dtype0*
_output_shapes
: 
�
Adam/dense_2/kernel/vVarHandleOp*
shape
: 
*&
shared_nameAdam/dense_2/kernel/v*
dtype0*
_output_shapes
: 
�
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
dtype0*
_output_shapes

: 
*(
_class
loc:@Adam/dense_2/kernel/v
~
Adam/dense_2/bias/vVarHandleOp*
shape:
*$
shared_nameAdam/dense_2/bias/v*
dtype0*
_output_shapes
: 
�
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*&
_class
loc:@Adam/dense_2/bias/v*
dtype0*
_output_shapes
:


NoOpNoOp
�;
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *�:
value�:B�: B�:
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
	keras_api
trainable_variables
regularization_losses

signatures
R
	variables
	keras_api
trainable_variables
regularization_losses
�

kernel
bias
_callable_losses
_eager_losses
	variables
	keras_api
trainable_variables
regularization_losses
�

kernel
bias
_callable_losses
_eager_losses
	variables
 	keras_api
!trainable_variables
"regularization_losses
{
#_callable_losses
$_eager_losses
%	variables
&	keras_api
'trainable_variables
(regularization_losses
{
)_callable_losses
*_eager_losses
+	variables
,	keras_api
-trainable_variables
.regularization_losses
�

/kernel
0bias
1_callable_losses
2_eager_losses
3	variables
4	keras_api
5trainable_variables
6regularization_losses
�

7kernel
8bias
9_callable_losses
:_eager_losses
;	variables
<	keras_api
=trainable_variables
>regularization_losses
�

?kernel
@bias
A_callable_losses
B_eager_losses
C	variables
D	keras_api
Etrainable_variables
Fregularization_losses
�
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratemsmtmumv/mw0mx7my8mz?m{@m|v}v~vv�/v�0v�7v�8v�?v�@v�
F
0
1
2
3
/4
05
76
87
?8
@9
y

	variables
regularization_losses

Llayers
Mmetrics
Nnon_trainable_variables
trainable_variables
F
0
1
2
3
/4
05
76
87
?8
@9
 
 
 
y
	variables
regularization_losses

Olayers
Pmetrics
Qnon_trainable_variables
trainable_variables
 
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
y
	variables
regularization_losses

Rlayers
Smetrics
Tnon_trainable_variables
trainable_variables

0
1
 
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
y
	variables
"regularization_losses

Ulayers
Vmetrics
Wnon_trainable_variables
!trainable_variables

0
1
 
 
 
 
y
%	variables
(regularization_losses

Xlayers
Ymetrics
Znon_trainable_variables
'trainable_variables
 
 
 
 
 
y
+	variables
.regularization_losses

[layers
\metrics
]non_trainable_variables
-trainable_variables
 
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

/0
01
y
3	variables
6regularization_losses

^layers
_metrics
`non_trainable_variables
5trainable_variables

/0
01
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

70
81
y
;	variables
>regularization_losses

alayers
bmetrics
cnon_trainable_variables
=trainable_variables

70
81
 
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
@1
y
C	variables
Fregularization_losses

dlayers
emetrics
fnon_trainable_variables
Etrainable_variables

?0
@1
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
8
0
1
2
3
4
5
6
7

g0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
	htotal
	icount
j
_fn_kwargs
k_updates
l	variables
m	keras_api
ntrainable_variables
oregularization_losses
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

h0
i1
y
l	variables
oregularization_losses

players
qmetrics
rnon_trainable_variables
ntrainable_variables
 
 
 
 

h0
i1
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*
dtype0*/
_output_shapes
:���������*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*,
f'R%
#__inference_signature_wrapper_67050*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������

O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*
_output_shapes
: *2
Tin+
)2'	*,
_gradient_op_typePartitionedCall-67188*'
f"R 
__inference__traced_save_67187*
Tout
2**
config_proto

CPU

GPU 2J 8
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v**
f%R#
!__inference__traced_restore_67311*
Tout
2**
config_proto

CPU

GPU 2J 8*1
Tin*
(2&*
_output_shapes
: *,
_gradient_op_typePartitionedCall-67312٧
�
I
-__inference_max_pooling2d_layer_call_fn_66807

inputs
identity�
PartitionedCallPartitionedCallinputs*J
_output_shapes8
6:4������������������������������������*
Tin
2*,
_gradient_op_typePartitionedCall-66804*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_66854

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�H@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:���������@*
T0"
identityIdentity:output:0*/
_input_shapes
:����������H::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
C
'__inference_flatten_layer_call_fn_66839

inputs
identity�
PartitionedCallPartitionedCallinputs*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_66835*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������H*,
_gradient_op_typePartitionedCall-66836a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������H"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_66865

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������@*
Tin
2*,
_gradient_op_typePartitionedCall-66860*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66854*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������H::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
%__inference_model_layer_call_fn_67025
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*,
_gradient_op_typePartitionedCall-67012*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_67011*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������
�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :	 :
 :' #
!
_user_specified_name	input_1: : : : : : : 
�=
�
 __inference__wrapped_model_66746
input_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource
identity��#model/conv2d/BiasAdd/ReadVariableOp�"model/conv2d/Conv2D/ReadVariableOp�%model/conv2d_1/BiasAdd/ReadVariableOp�$model/conv2d_1/Conv2D/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:��������� �
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� r
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @�
model/conv2d_1/Conv2DConv2Dmodel/conv2d/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:���������@�
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:���������@*
T0v
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
model/max_pooling2d/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*
ksize
*
paddingVALID*/
_output_shapes
:���������@*
strides
g
model/flatten/ShapeShape$model/max_pooling2d/MaxPool:output:0*
_output_shapes
:*
T0k
!model/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:m
#model/flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:m
#model/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
model/flatten/strided_sliceStridedSlicemodel/flatten/Shape:output:0*model/flatten/strided_slice/stack:output:0,model/flatten/strided_slice/stack_1:output:0,model/flatten/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: h
model/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
����������
model/flatten/Reshape/shapePack$model/flatten/strided_slice:output:0&model/flatten/Reshape/shape/1:output:0*
T0*
N*
_output_shapes
:�
model/flatten/ReshapeReshape$model/max_pooling2d/MaxPool:output:0$model/flatten/Reshape/shape:output:0*(
_output_shapes
:����������H*
T0�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�H@�
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ �
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:��������� *
T0�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:��������� *
T0l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*'
_output_shapes
:��������� *
T0�
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

: 
�
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������
*
T0�
IdentityIdentitymodel/dense_2/BiasAdd:output:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp: :	 :
 :' #
!
_user_specified_name	input_1: : : : : : : 
��
�
!__inference__traced_restore_67311
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias%
!assignvariableop_8_dense_2_kernel#
assignvariableop_9_dense_2_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count,
(assignvariableop_17_adam_conv2d_kernel_m*
&assignvariableop_18_adam_conv2d_bias_m.
*assignvariableop_19_adam_conv2d_1_kernel_m,
(assignvariableop_20_adam_conv2d_1_bias_m+
'assignvariableop_21_adam_dense_kernel_m)
%assignvariableop_22_adam_dense_bias_m-
)assignvariableop_23_adam_dense_1_kernel_m+
'assignvariableop_24_adam_dense_1_bias_m-
)assignvariableop_25_adam_dense_2_kernel_m+
'assignvariableop_26_adam_dense_2_bias_m,
(assignvariableop_27_adam_conv2d_kernel_v*
&assignvariableop_28_adam_conv2d_bias_v.
*assignvariableop_29_adam_conv2d_1_kernel_v,
(assignvariableop_30_adam_conv2d_1_bias_v+
'assignvariableop_31_adam_dense_kernel_v)
%assignvariableop_32_adam_dense_bias_v-
)assignvariableop_33_adam_dense_1_kernel_v+
'assignvariableop_34_adam_dense_1_bias_v-
)assignvariableop_35_adam_dense_2_kernel_v+
'assignvariableop_36_adam_dense_2_bias_v
identity_38��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:%*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE�
RestoreV2/shape_and_slicesConst"/device:CPU:0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:%�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:}
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
dtype0	*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:{
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:{
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv2d_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv2d_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_1_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_1_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_2_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_2_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_vIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_vIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_2_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_2_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_38Identity_38:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::2
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9: : : : : : : : : : : : : : :  :! :" :# :$ :% :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : 
�!
�
@__inference_model_layer_call_and_return_conditional_losses_66925
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:��������� *,
_gradient_op_typePartitionedCall-66764*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_66763�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*/
_output_shapes
:���������@*
Tin
2*,
_gradient_op_typePartitionedCall-66787*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786*
Tout
2**
config_proto

CPU

GPU 2J 8�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@*,
_gradient_op_typePartitionedCall-66804*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������H*,
_gradient_op_typePartitionedCall-66836*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_66835*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66860*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66854*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������@�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66886*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66885*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:��������� �
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*'
_output_shapes
:���������
*
Tin
2*,
_gradient_op_typePartitionedCall-66913*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_66907*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*'
_output_shapes
:���������
*
T0"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:
 :' #
!
_user_specified_name	input_1: : : : : : : : :	 
�
�
B__inference_dense_2_layer_call_and_return_conditional_losses_66907

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

: 
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�!
�
@__inference_model_layer_call_and_return_conditional_losses_66948
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_66763*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:��������� *
Tin
2*,
_gradient_op_typePartitionedCall-66764�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������@*
Tin
2*,
_gradient_op_typePartitionedCall-66787*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-66804*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������@*
Tin
2�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:����������H*
Tin
2*,
_gradient_op_typePartitionedCall-66836*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_66835*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66860*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66854*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������@�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66886*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66885*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:��������� �
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66913*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_66907*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������
�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 
�
�
'__inference_dense_2_layer_call_fn_66918

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_66907*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������
*,
_gradient_op_typePartitionedCall-66913�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�!
�
@__inference_model_layer_call_and_return_conditional_losses_67011

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*/
_output_shapes
:��������� *,
_gradient_op_typePartitionedCall-66764*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_66763*
Tout
2**
config_proto

CPU

GPU 2J 8�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@*,
_gradient_op_typePartitionedCall-66787*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-66804*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������@*
Tin
2�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������H*,
_gradient_op_typePartitionedCall-66836*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_66835�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:���������@*,
_gradient_op_typePartitionedCall-66860*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66854*
Tout
2**
config_proto

CPU

GPU 2J 8�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66886*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66885*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:��������� *
Tin
2�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������
*
Tin
2*,
_gradient_op_typePartitionedCall-66913*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_66907�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:
 :& "
 
_user_specified_nameinputs: : : : : : : : :	 
�H
�
__inference__traced_save_67187
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_62e2a0c501a642a0ae487b3f16d6f59d/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:%�
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:%*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B �
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : @:@:	�H@:@:@ : : 
:
: : : : : : : : : : @:@:	�H@:@:@ : : 
:
: : : @:@:	�H@:@:@ : : 
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : 
�
�
&__inference_conv2d_layer_call_fn_66769

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+��������������������������� *,
_gradient_op_typePartitionedCall-66764*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_66763*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*A
_output_shapes/
-:+���������������������������@*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_66763

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+��������������������������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+��������������������������� *
T0�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�!
�
@__inference_model_layer_call_and_return_conditional_losses_66972

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:��������� *
Tin
2*,
_gradient_op_typePartitionedCall-66764*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_66763�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������@*
Tin
2*,
_gradient_op_typePartitionedCall-66787*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786*
Tout
2�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������@*,
_gradient_op_typePartitionedCall-66804*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������H*,
_gradient_op_typePartitionedCall-66836*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_66835*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������@*
Tin
2*,
_gradient_op_typePartitionedCall-66860*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66854�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:��������� *,
_gradient_op_typePartitionedCall-66886*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66885*
Tout
2�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66913*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_66907*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������
*
Tin
2�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:
 :& "
 
_user_specified_nameinputs: : : : : : : : :	 
�
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803

inputs
identity�
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�	
^
B__inference_flatten_layer_call_and_return_conditional_losses_66835

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0Z
Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
T0*
N*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������HY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������H"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_66986
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*,
_gradient_op_typePartitionedCall-66973*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_66972*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:���������
*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 
�
�
#__inference_signature_wrapper_67050
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*,
_gradient_op_typePartitionedCall-67037*)
f$R"
 __inference__wrapped_model_66746*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������
�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 :' #
!
_user_specified_name	input_1: 
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_66885

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
'__inference_dense_1_layer_call_fn_66891

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-66886*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66885*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:��������� �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
(__inference_conv2d_1_layer_call_fn_66792

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+���������������������������@*,
_gradient_op_typePartitionedCall-66787�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: "7L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������;
dense_20
StatefulPartitionedCall:0���������
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�>
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
	keras_api
trainable_variables
regularization_losses

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�:
_tf_keras_model�:{"input_spec": null, "name": "model", "config": {"name": "model", "layers": [{"inbound_nodes": [], "config": {"sparse": false, "name": "input_1", "batch_input_shape": [null, 28, 28, 1], "dtype": "float32"}, "class_name": "InputLayer", "name": "input_1"}, {"inbound_nodes": [[["input_1", 0, 0, {}]]], "config": {"activation": "relu", "name": "conv2d", "strides": [1, 1], "dilation_rate": [1, 1], "use_bias": true, "bias_constraint": null, "filters": 32, "bias_regularizer": null, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "kernel_size": [3, 3], "kernel_regularizer": null, "kernel_constraint": null, "padding": "valid", "activity_regularizer": null, "trainable": true, "dtype": "float32", "data_format": "channels_last", "bias_initializer": {"config": {}, "class_name": "Zeros"}}, "class_name": "Conv2D", "name": "conv2d"}, {"inbound_nodes": [[["conv2d", 0, 0, {}]]], "config": {"activation": "relu", "name": "conv2d_1", "strides": [1, 1], "dilation_rate": [1, 1], "use_bias": true, "bias_constraint": null, "filters": 64, "bias_regularizer": null, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "kernel_size": [3, 3], "kernel_regularizer": null, "kernel_constraint": null, "padding": "valid", "activity_regularizer": null, "trainable": true, "dtype": "float32", "data_format": "channels_last", "bias_initializer": {"config": {}, "class_name": "Zeros"}}, "class_name": "Conv2D", "name": "conv2d_1"}, {"inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "config": {"padding": "valid", "name": "max_pooling2d", "strides": [2, 2], "trainable": true, "dtype": "float32", "data_format": "channels_last", "pool_size": [2, 2]}, "class_name": "MaxPooling2D", "name": "max_pooling2d"}, {"inbound_nodes": [[["max_pooling2d", 0, 0, {}]]], "config": {"trainable": true, "name": "flatten", "dtype": "float32", "data_format": "channels_last"}, "class_name": "Flatten", "name": "flatten"}, {"inbound_nodes": [[["flatten", 0, 0, {}]]], "config": {"activation": "relu", "name": "dense", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "kernel_constraint": null, "units": 64, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_regularizer": null, "use_bias": true, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "class_name": "Dense", "name": "dense"}, {"inbound_nodes": [[["dense", 0, 0, {}]]], "config": {"activation": "relu", "name": "dense_1", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "kernel_constraint": null, "units": 32, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_regularizer": null, "use_bias": true, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "class_name": "Dense", "name": "dense_1"}, {"inbound_nodes": [[["dense_1", 0, 0, {}]]], "config": {"activation": "linear", "name": "dense_2", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "kernel_constraint": null, "units": 10, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_regularizer": null, "use_bias": true, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "class_name": "Dense", "name": "dense_2"}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "training_config": {"sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"config": {"name": "Adam", "beta_1": 0.8999999761581421, "learning_rate": 0.003000000026077032, "beta_2": 0.9990000128746033, "decay": 0.0, "epsilon": 1e-07, "amsgrad": false}, "class_name": "Adam"}, "loss": "loss_func", "metrics": ["accuracy"], "loss_weights": null}, "backend": "tensorflow", "activity_regularizer": null, "dtype": null, "class_name": "Model", "model_config": {"config": {"name": "model", "layers": [{"inbound_nodes": [], "config": {"dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "name": "input_1", "sparse": false}, "class_name": "InputLayer", "name": "input_1"}, {"inbound_nodes": [[["input_1", 0, 0, {}]]], "config": {"activation": "relu", "name": "conv2d", "dilation_rate": [1, 1], "strides": [1, 1], "kernel_constraint": null, "bias_constraint": null, "filters": 32, "activity_regularizer": null, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "kernel_size": [3, 3], "kernel_regularizer": null, "use_bias": true, "padding": "valid", "bias_regularizer": null, "trainable": true, "dtype": "float32", "data_format": "channels_last", "bias_initializer": {"config": {}, "class_name": "Zeros"}}, "class_name": "Conv2D", "name": "conv2d"}, {"inbound_nodes": [[["conv2d", 0, 0, {}]]], "config": {"activation": "relu", "name": "conv2d_1", "dilation_rate": [1, 1], "strides": [1, 1], "kernel_constraint": null, "bias_constraint": null, "filters": 64, "activity_regularizer": null, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "kernel_size": [3, 3], "kernel_regularizer": null, "use_bias": true, "padding": "valid", "bias_regularizer": null, "trainable": true, "dtype": "float32", "data_format": "channels_last", "bias_initializer": {"config": {}, "class_name": "Zeros"}}, "class_name": "Conv2D", "name": "conv2d_1"}, {"inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "config": {"padding": "valid", "name": "max_pooling2d", "strides": [2, 2], "trainable": true, "dtype": "float32", "data_format": "channels_last", "pool_size": [2, 2]}, "class_name": "MaxPooling2D", "name": "max_pooling2d"}, {"inbound_nodes": [[["max_pooling2d", 0, 0, {}]]], "config": {"trainable": true, "name": "flatten", "dtype": "float32", "data_format": "channels_last"}, "class_name": "Flatten", "name": "flatten"}, {"inbound_nodes": [[["flatten", 0, 0, {}]]], "config": {"activation": "relu", "name": "dense", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "use_bias": true, "units": 64, "bias_constraint": null, "activity_regularizer": null, "dtype": "float32", "kernel_regularizer": null, "kernel_constraint": null, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "class_name": "Dense", "name": "dense"}, {"inbound_nodes": [[["dense", 0, 0, {}]]], "config": {"activation": "relu", "name": "dense_1", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "use_bias": true, "units": 32, "bias_constraint": null, "activity_regularizer": null, "dtype": "float32", "kernel_regularizer": null, "kernel_constraint": null, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "class_name": "Dense", "name": "dense_1"}, {"inbound_nodes": [[["dense_1", 0, 0, {}]]], "config": {"activation": "linear", "name": "dense_2", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "use_bias": true, "units": 10, "bias_constraint": null, "activity_regularizer": null, "dtype": "float32", "kernel_regularizer": null, "kernel_constraint": null, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "class_name": "Dense", "name": "dense_2"}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "class_name": "Model"}, "batch_input_shape": null, "expects_training_arg": true, "trainable": true, "keras_version": "2.2.4-tf"}
�
	variables
	keras_api
trainable_variables
regularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": null, "batch_input_shape": [null, 28, 28, 1], "name": "input_1", "config": {"sparse": false, "name": "input_1", "batch_input_shape": [null, 28, 28, 1], "dtype": "float32"}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "InputLayer"}
�

kernel
bias
_callable_losses
_eager_losses
	variables
	keras_api
trainable_variables
regularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": {"config": {"axes": {"-1": 1}, "max_ndim": null, "ndim": 4, "shape": null, "dtype": null, "min_ndim": null}, "class_name": "InputSpec"}, "batch_input_shape": null, "name": "conv2d", "config": {"activation": "relu", "name": "conv2d", "strides": [1, 1], "dilation_rate": [1, 1], "use_bias": true, "bias_constraint": null, "filters": 32, "bias_regularizer": null, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "kernel_size": [3, 3], "kernel_regularizer": null, "kernel_constraint": null, "padding": "valid", "activity_regularizer": null, "trainable": true, "dtype": "float32", "data_format": "channels_last", "bias_initializer": {"config": {}, "class_name": "Zeros"}}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "Conv2D"}
�

kernel
bias
_callable_losses
_eager_losses
	variables
 	keras_api
!trainable_variables
"regularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": {"config": {"axes": {"-1": 32}, "max_ndim": null, "ndim": 4, "shape": null, "dtype": null, "min_ndim": null}, "class_name": "InputSpec"}, "batch_input_shape": null, "name": "conv2d_1", "config": {"activation": "relu", "name": "conv2d_1", "strides": [1, 1], "dilation_rate": [1, 1], "use_bias": true, "bias_constraint": null, "filters": 64, "bias_regularizer": null, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "kernel_size": [3, 3], "kernel_regularizer": null, "kernel_constraint": null, "padding": "valid", "activity_regularizer": null, "trainable": true, "dtype": "float32", "data_format": "channels_last", "bias_initializer": {"config": {}, "class_name": "Zeros"}}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "Conv2D"}
�
#_callable_losses
$_eager_losses
%	variables
&	keras_api
'trainable_variables
(regularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": {"config": {"axes": {}, "max_ndim": null, "ndim": 4, "shape": null, "dtype": null, "min_ndim": null}, "class_name": "InputSpec"}, "batch_input_shape": null, "name": "max_pooling2d", "config": {"padding": "valid", "name": "max_pooling2d", "strides": [2, 2], "trainable": true, "dtype": "float32", "data_format": "channels_last", "pool_size": [2, 2]}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "MaxPooling2D"}
�
)_callable_losses
*_eager_losses
+	variables
,	keras_api
-trainable_variables
.regularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": {"config": {"axes": {}, "max_ndim": null, "ndim": null, "shape": null, "dtype": null, "min_ndim": 1}, "class_name": "InputSpec"}, "batch_input_shape": null, "name": "flatten", "config": {"trainable": true, "name": "flatten", "dtype": "float32", "data_format": "channels_last"}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "Flatten"}
�

/kernel
0bias
1_callable_losses
2_eager_losses
3	variables
4	keras_api
5trainable_variables
6regularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": {"config": {"axes": {"-1": 9216}, "max_ndim": null, "ndim": null, "shape": null, "dtype": null, "min_ndim": 2}, "class_name": "InputSpec"}, "batch_input_shape": null, "name": "dense", "config": {"activation": "relu", "name": "dense", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "kernel_constraint": null, "units": 64, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_regularizer": null, "use_bias": true, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "Dense"}
�

7kernel
8bias
9_callable_losses
:_eager_losses
;	variables
<	keras_api
=trainable_variables
>regularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": {"config": {"axes": {"-1": 64}, "max_ndim": null, "ndim": null, "shape": null, "dtype": null, "min_ndim": 2}, "class_name": "InputSpec"}, "batch_input_shape": null, "name": "dense_1", "config": {"activation": "relu", "name": "dense_1", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "kernel_constraint": null, "units": 32, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_regularizer": null, "use_bias": true, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "Dense"}
�

?kernel
@bias
A_callable_losses
B_eager_losses
C	variables
D	keras_api
Etrainable_variables
Fregularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": {"config": {"axes": {"-1": 32}, "max_ndim": null, "ndim": null, "shape": null, "dtype": null, "min_ndim": 2}, "class_name": "InputSpec"}, "batch_input_shape": null, "name": "dense_2", "config": {"activation": "linear", "name": "dense_2", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "kernel_constraint": null, "units": 10, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_regularizer": null, "use_bias": true, "bias_regularizer": null, "trainable": true, "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "Dense"}
�
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_ratemsmtmumv/mw0mx7my8mz?m{@m|v}v~vv�/v�0v�7v�8v�?v�@v�"
	optimizer
f
0
1
2
3
/4
05
76
87
?8
@9"
trackable_list_wrapper
�

	variables
regularization_losses

Llayers
Mmetrics
Nnon_trainable_variables
trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�_default_save_signature
�__call__"
_generic_user_object
f
0
1
2
3
/4
05
76
87
?8
@9"
trackable_list_wrapper
 "
trackable_list_wrapper
-
�serving_default"
signature_map
 "
trackable_list_wrapper
�
	variables
regularization_losses

Olayers
Pmetrics
Qnon_trainable_variables
trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
regularization_losses

Rlayers
Smetrics
Tnon_trainable_variables
trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
):' @2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
"regularization_losses

Ulayers
Vmetrics
Wnon_trainable_variables
!trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
%	variables
(regularization_losses

Xlayers
Ymetrics
Znon_trainable_variables
'trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
+	variables
.regularization_losses

[layers
\metrics
]non_trainable_variables
-trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	�H@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
3	variables
6regularization_losses

^layers
_metrics
`non_trainable_variables
5trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 :@ 2dense_1/kernel
: 2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�
;	variables
>regularization_losses

alayers
bmetrics
cnon_trainable_variables
=trainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 : 
2dense_2/kernel
:
2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
C	variables
Fregularization_losses

dlayers
emetrics
fnon_trainable_variables
Etrainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	htotal
	icount
j
_fn_kwargs
k_updates
l	variables
m	keras_api
ntrainable_variables
oregularization_losses
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"input_spec": null, "batch_input_shape": null, "name": "accuracy", "config": {"name": "accuracy", "dtype": "float32"}, "expects_training_arg": false, "trainable": true, "activity_regularizer": null, "dtype": "float32", "class_name": "MeanMetricWrapper"}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
�
l	variables
oregularization_losses

players
qmetrics
rnon_trainable_variables
ntrainable_variables
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses
�__call__"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
$:"	�H@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@ 2Adam/dense_1/kernel/m
: 2Adam/dense_1/bias/m
%:# 
2Adam/dense_2/kernel/m
:
2Adam/dense_2/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
$:"	�H@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@ 2Adam/dense_1/kernel/v
: 2Adam/dense_1/bias/v
%:# 
2Adam/dense_2/kernel/v
:
2Adam/dense_2/bias/v
�2�
@__inference_model_layer_call_and_return_conditional_losses_66948
@__inference_model_layer_call_and_return_conditional_losses_66925
@__inference_model_layer_call_and_return_conditional_losses_67011
@__inference_model_layer_call_and_return_conditional_losses_66972�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
 __inference__wrapped_model_66746�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������
�2�
%__inference_model_layer_call_fn_67025
%__inference_model_layer_call_fn_66986�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_conv2d_layer_call_and_return_conditional_losses_66763�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
&__inference_conv2d_layer_call_fn_66769�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
(__inference_conv2d_1_layer_call_fn_66792�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_max_pooling2d_layer_call_fn_66807�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
B__inference_flatten_layer_call_and_return_conditional_losses_66835�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_flatten_layer_call_fn_66839�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_66854�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_66865�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_66885�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_1_layer_call_fn_66891�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_2_layer_call_and_return_conditional_losses_66907�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_2_layer_call_fn_66918�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
#__inference_signature_wrapper_67050input_1
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
@__inference_model_layer_call_and_return_conditional_losses_66925q
/078?@<�9
2�/
)�&
input_1���������
p 
� "%�"
�
0���������

� �
B__inference_flatten_layer_call_and_return_conditional_losses_66835a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������H
� �
&__inference_conv2d_layer_call_fn_66769�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
#__inference_signature_wrapper_67050�
/078?@C�@
� 
9�6
4
input_1)�&
input_1���������"1�.
,
dense_2!�
dense_2���������
�
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_66803�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� z
'__inference_dense_1_layer_call_fn_66891O78/�,
%�"
 �
inputs���������@
� "���������� y
%__inference_dense_layer_call_fn_66865P/00�-
&�#
!�
inputs����������H
� "����������@�
@__inference_model_layer_call_and_return_conditional_losses_66948q
/078?@<�9
2�/
)�&
input_1���������
p
� "%�"
�
0���������

� �
%__inference_model_layer_call_fn_67025d
/078?@<�9
2�/
)�&
input_1���������
p
� "����������
�
B__inference_dense_1_layer_call_and_return_conditional_losses_66885\78/�,
%�"
 �
inputs���������@
� "%�"
�
0��������� 
� �
B__inference_dense_2_layer_call_and_return_conditional_losses_66907\?@/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������

� �
%__inference_model_layer_call_fn_66986d
/078?@<�9
2�/
)�&
input_1���������
p 
� "����������
�
A__inference_conv2d_layer_call_and_return_conditional_losses_66763�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
@__inference_model_layer_call_and_return_conditional_losses_67011p
/078?@;�8
1�.
(�%
inputs���������
p
� "%�"
�
0���������

� �
 __inference__wrapped_model_66746y
/078?@8�5
.�+
)�&
input_1���������
� "1�.
,
dense_2!�
dense_2���������
z
'__inference_dense_2_layer_call_fn_66918O?@/�,
%�"
 �
inputs��������� 
� "����������
�
@__inference_model_layer_call_and_return_conditional_losses_66972p
/078?@;�8
1�.
(�%
inputs���������
p 
� "%�"
�
0���������

� �
-__inference_max_pooling2d_layer_call_fn_66807�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
@__inference_dense_layer_call_and_return_conditional_losses_66854]/00�-
&�#
!�
inputs����������H
� "%�"
�
0���������@
� 
'__inference_flatten_layer_call_fn_66839T7�4
-�*
(�%
inputs���������@
� "�����������H�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_66786�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
(__inference_conv2d_1_layer_call_fn_66792�I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@