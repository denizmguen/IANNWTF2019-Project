��
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
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108��	
�
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_9/gamma
�
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_9/beta
�
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_9/moving_mean
�
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_9/moving_variance
�
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:(*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:(*
dtype0
�
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*-
shared_namebatch_normalization_10/gamma
�
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:(*
dtype0
�
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*,
shared_namebatch_normalization_10/beta
�
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:(*
dtype0
�
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*3
shared_name$"batch_normalization_10/moving_mean
�
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:(*
dtype0
�
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*7
shared_name(&batch_normalization_10/moving_variance
�
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:(*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:)* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:)*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
�'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�&
value�&B�& B�&
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
�
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
�
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
a
(	constants
)regularization_losses
*trainable_variables
+	variables
,	keras_api
 
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
 
 
F
0
1
2
3
 4
!5
16
27
78
89
f
0
1
2
3
4
5
 6
!7
"8
#9
110
211
712
813
�
regularization_losses
trainable_variables

=layers
>metrics
?layer_regularization_losses
@non_trainable_variables
	variables
 
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
�
regularization_losses
trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables
	variables
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
	variables
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
"2
#3
�
$regularization_losses
%trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Lnon_trainable_variables
&	variables
 
 
 
 
�
)regularization_losses
*trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Pnon_trainable_variables
+	variables
 
 
 
�
-regularization_losses
.trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tnon_trainable_variables
/	variables
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
�
3regularization_losses
4trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xnon_trainable_variables
5	variables
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
�
9regularization_losses
:trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\non_trainable_variables
;	variables
?
0
1
2
3
4
5
6
7
	8

]0
 

0
1
"2
#3
 
 
 

0
1
 
 
 
 
 
 
 

"0
#1
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
x
	^total
	_count
`
_fn_kwargs
aregularization_losses
btrainable_variables
c	variables
d	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

^0
_1
�
aregularization_losses
btrainable_variables

elayers
fmetrics
glayer_regularization_losses
hnon_trainable_variables
c	variables
 
 
 

^0
_1
z
serving_default_input_8Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_9Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8serving_default_input_9%batch_normalization_9/moving_variancebatch_normalization_9/gamma!batch_normalization_9/moving_meanbatch_normalization_9/betadense_12/kerneldense_12/bias&batch_normalization_10/moving_variancebatch_normalization_10/gamma"batch_normalization_10/moving_meanbatch_normalization_10/betadense_13/kerneldense_13/biasdense_14/kerneldense_14/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_476965
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_477562
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancedense_12/kerneldense_12/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancedense_13/kerneldense_13/biasdense_14/kerneldense_14/biastotalcount*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_477622��	
�
�
(__inference_model_4_layer_call_fn_476822
input_8
input_9"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8input_9statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4768052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_8:'#
!
_user_specified_name	input_9
�2
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_477252

inputs
assignmovingavg_477225
assignmovingavg_1_477232)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/477225*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*)
_class
loc:@AssignMovingAvg/477225*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_477225*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/477225*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*)
_class
loc:@AssignMovingAvg/477225*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_477225AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/477225*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/477232*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*+
_class!
loc:@AssignMovingAvg_1/477232*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_477232*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/477232*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*+
_class!
loc:@AssignMovingAvg_1/477232*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_477232AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/477232*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_dense_12_layer_call_fn_477310

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4766382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
Z
.__inference_concatenate_3_layer_call_fn_477453
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������)*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4766932
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������)2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������(:���������:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�2
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_477389

inputs
assignmovingavg_477362
assignmovingavg_1_477369)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:(2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������(2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/477362*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*)
_class
loc:@AssignMovingAvg/477362*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_477362*
_output_shapes
:(*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/477362*
_output_shapes
:(2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*)
_class
loc:@AssignMovingAvg/477362*
_output_shapes
:(2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_477362AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/477362*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/477369*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*+
_class!
loc:@AssignMovingAvg_1/477369*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_477369*
_output_shapes
:(*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/477369*
_output_shapes
:(2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*+
_class!
loc:@AssignMovingAvg_1/477369*
_output_shapes
:(2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_477369AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/477369*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������(::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
D__inference_dense_14_layer_call_and_return_conditional_losses_476736

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
D__inference_dense_13_layer_call_and_return_conditional_losses_477464

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������)::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
D__inference_dense_12_layer_call_and_return_conditional_losses_476638

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_10_layer_call_fn_477421

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_4765612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������(::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_model_4_layer_call_fn_476868
input_8
input_9"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8input_9statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4768512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_8:'#
!
_user_specified_name	input_9
�
�
7__inference_batch_normalization_10_layer_call_fn_477430

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_4765932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������(::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
j
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_476678

inputs
identitya
Relu_5Reluinputs*
T0*
_cloned(*'
_output_shapes
:���������(2
Relu_5h
IdentityIdentityRelu_5:activations:0*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������(:& "
 
_user_specified_nameinputs
�
�
)__inference_dense_13_layer_call_fn_477471

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4767132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������)::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�*
�
C__inference_model_4_layer_call_and_return_conditional_losses_476775
input_8
input_98
4batch_normalization_9_statefulpartitionedcall_args_18
4batch_normalization_9_statefulpartitionedcall_args_28
4batch_normalization_9_statefulpartitionedcall_args_38
4batch_normalization_9_statefulpartitionedcall_args_4+
'dense_12_statefulpartitionedcall_args_1+
'dense_12_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_19
5batch_normalization_10_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_39
5batch_normalization_10_statefulpartitionedcall_args_4+
'dense_13_statefulpartitionedcall_args_1+
'dense_13_statefulpartitionedcall_args_2+
'dense_14_statefulpartitionedcall_args_1+
'dense_14_statefulpartitionedcall_args_2
identity��.batch_normalization_10/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCallinput_84batch_normalization_9_statefulpartitionedcall_args_14batch_normalization_9_statefulpartitionedcall_args_24batch_normalization_9_statefulpartitionedcall_args_34batch_normalization_9_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_4764452/
-batch_normalization_9/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0'dense_12_statefulpartitionedcall_args_1'dense_12_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4766382"
 dense_12/StatefulPartitionedCall�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:05batch_normalization_10_statefulpartitionedcall_args_15batch_normalization_10_statefulpartitionedcall_args_25batch_normalization_10_statefulpartitionedcall_args_35batch_normalization_10_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_47659320
.batch_normalization_10/StatefulPartitionedCall�
"tf_op_layer_Relu_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_4766782$
"tf_op_layer_Relu_5/PartitionedCall�
concatenate_3/PartitionedCallPartitionedCall+tf_op_layer_Relu_5/PartitionedCall:output:0input_9*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������)*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4766932
concatenate_3/PartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0'dense_13_statefulpartitionedcall_args_1'dense_13_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4767132"
 dense_13/StatefulPartitionedCall�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0'dense_14_statefulpartitionedcall_args_1'dense_14_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_4767362"
 dense_14/StatefulPartitionedCall�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:' #
!
_user_specified_name	input_8:'#
!
_user_specified_name	input_9
�*
�
C__inference_model_4_layer_call_and_return_conditional_losses_476851

inputs
inputs_18
4batch_normalization_9_statefulpartitionedcall_args_18
4batch_normalization_9_statefulpartitionedcall_args_28
4batch_normalization_9_statefulpartitionedcall_args_38
4batch_normalization_9_statefulpartitionedcall_args_4+
'dense_12_statefulpartitionedcall_args_1+
'dense_12_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_19
5batch_normalization_10_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_39
5batch_normalization_10_statefulpartitionedcall_args_4+
'dense_13_statefulpartitionedcall_args_1+
'dense_13_statefulpartitionedcall_args_2+
'dense_14_statefulpartitionedcall_args_1+
'dense_14_statefulpartitionedcall_args_2
identity��.batch_normalization_10/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCallinputs4batch_normalization_9_statefulpartitionedcall_args_14batch_normalization_9_statefulpartitionedcall_args_24batch_normalization_9_statefulpartitionedcall_args_34batch_normalization_9_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_4764452/
-batch_normalization_9/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0'dense_12_statefulpartitionedcall_args_1'dense_12_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4766382"
 dense_12/StatefulPartitionedCall�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:05batch_normalization_10_statefulpartitionedcall_args_15batch_normalization_10_statefulpartitionedcall_args_25batch_normalization_10_statefulpartitionedcall_args_35batch_normalization_10_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_47659320
.batch_normalization_10/StatefulPartitionedCall�
"tf_op_layer_Relu_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_4766782$
"tf_op_layer_Relu_5/PartitionedCall�
concatenate_3/PartitionedCallPartitionedCall+tf_op_layer_Relu_5/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������)*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4766932
concatenate_3/PartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0'dense_13_statefulpartitionedcall_args_1'dense_13_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4767132"
 dense_13/StatefulPartitionedCall�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0'dense_14_statefulpartitionedcall_args_1'dense_14_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_4767362"
 dense_14/StatefulPartitionedCall�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
(__inference_model_4_layer_call_fn_477173
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4768512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
Q
3__inference_tf_op_layer_Relu_5_layer_call_fn_477440
inputs_0
identity�
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_4766782
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������(:( $
"
_user_specified_name
inputs/0
�
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_476693

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������)2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������)2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������(:���������:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
C__inference_model_4_layer_call_and_return_conditional_losses_477067
inputs_0
inputs_10
,batch_normalization_9_assignmovingavg_4769802
.batch_normalization_9_assignmovingavg_1_476987?
;batch_normalization_9_batchnorm_mul_readvariableop_resource;
7batch_normalization_9_batchnorm_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource1
-batch_normalization_10_assignmovingavg_4770233
/batch_normalization_10_assignmovingavg_1_477030@
<batch_normalization_10_batchnorm_mul_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource
identity��:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_10/AssignMovingAvg/ReadVariableOp�<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_10/batchnorm/ReadVariableOp�3batch_normalization_10/batchnorm/mul/ReadVariableOp�9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_9/AssignMovingAvg/ReadVariableOp�;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_9/batchnorm/ReadVariableOp�2batch_normalization_9/batchnorm/mul/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices�
"batch_normalization_9/moments/meanMeaninputs_0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_9/moments/mean�
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_9/moments/StopGradient�
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferenceinputs_03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������21
/batch_normalization_9/moments/SquaredDifference�
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices�
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_9/moments/variance�
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze�
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1�
+batch_normalization_9/AssignMovingAvg/decayConst*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/476980*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_9/AssignMovingAvg/decay�
*batch_normalization_9/AssignMovingAvg/CastCast4batch_normalization_9/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/476980*
_output_shapes
: 2,
*batch_normalization_9/AssignMovingAvg/Cast�
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_9_assignmovingavg_476980*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp�
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/476980*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/sub�
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:0.batch_normalization_9/AssignMovingAvg/Cast:y:0*
T0*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/476980*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul�
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_9_assignmovingavg_476980-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*?
_class5
31loc:@batch_normalization_9/AssignMovingAvg/476980*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_9/AssignMovingAvg_1/decayConst*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/476987*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_9/AssignMovingAvg_1/decay�
,batch_normalization_9/AssignMovingAvg_1/CastCast6batch_normalization_9/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/476987*
_output_shapes
: 2.
,batch_normalization_9/AssignMovingAvg_1/Cast�
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_9_assignmovingavg_1_476987*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/476987*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/sub�
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:00batch_normalization_9/AssignMovingAvg_1/Cast:y:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/476987*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul�
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_9_assignmovingavg_1_476987/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg_1/476987*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt�
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_9/batchnorm/mul/ReadVariableOp�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Mulinputs_0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2�
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_9/batchnorm/ReadVariableOp�
#batch_normalization_9/batchnorm/subSub6batch_normalization_9/batchnorm/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/add_1�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMul)batch_normalization_9/batchnorm/add_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_12/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indices�
#batch_normalization_10/moments/meanMeandense_12/BiasAdd:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2%
#batch_normalization_10/moments/mean�
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes

:(2-
+batch_normalization_10/moments/StopGradient�
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_12/BiasAdd:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������(22
0batch_normalization_10/moments/SquaredDifference�
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices�
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2)
'batch_normalization_10/moments/variance�
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze�
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1�
,batch_normalization_10/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/477023*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_10/AssignMovingAvg/decay�
+batch_normalization_10/AssignMovingAvg/CastCast5batch_normalization_10/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/477023*
_output_shapes
: 2-
+batch_normalization_10/AssignMovingAvg/Cast�
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_10_assignmovingavg_477023*
_output_shapes
:(*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp�
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/477023*
_output_shapes
:(2,
*batch_normalization_10/AssignMovingAvg/sub�
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:0/batch_normalization_10/AssignMovingAvg/Cast:y:0*
T0*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/477023*
_output_shapes
:(2,
*batch_normalization_10/AssignMovingAvg/mul�
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_10_assignmovingavg_477023.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_10/AssignMovingAvg/477023*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_10/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/477030*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_10/AssignMovingAvg_1/decay�
-batch_normalization_10/AssignMovingAvg_1/CastCast7batch_normalization_10/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/477030*
_output_shapes
: 2/
-batch_normalization_10/AssignMovingAvg_1/Cast�
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_10_assignmovingavg_1_477030*
_output_shapes
:(*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/477030*
_output_shapes
:(2.
,batch_normalization_10/AssignMovingAvg_1/sub�
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:01batch_normalization_10/AssignMovingAvg_1/Cast:y:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/477030*
_output_shapes
:(2.
,batch_normalization_10/AssignMovingAvg_1/mul�
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_10_assignmovingavg_1_4770300batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg_1/477030*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_10/batchnorm/Rsqrt�
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOp�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_12/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_10/batchnorm/mul_2�
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype021
/batch_normalization_10/batchnorm/ReadVariableOp�
$batch_normalization_10/batchnorm/subSub7batch_normalization_10/batchnorm/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_10/batchnorm/add_1�
tf_op_layer_Relu_5/Relu_5Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*
_cloned(*'
_output_shapes
:���������(2
tf_op_layer_Relu_5/Relu_5x
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis�
concatenate_3/concatConcatV2'tf_op_layer_Relu_5/Relu_5:activations:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������)2
concatenate_3/concat�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:)*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulconcatenate_3/concat:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_13/Relu�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SigmoidSigmoiddense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Sigmoid�
IdentityIdentitydense_14/Sigmoid:y:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_10/AssignMovingAvg/ReadVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_10/batchnorm/ReadVariableOp4^batch_normalization_10/batchnorm/mul/ReadVariableOp:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_9/AssignMovingAvg/ReadVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp3^batch_normalization_9/batchnorm/mul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�o
�
!__inference__wrapped_model_476304
input_8
input_9C
?model_4_batch_normalization_9_batchnorm_readvariableop_resourceG
Cmodel_4_batch_normalization_9_batchnorm_mul_readvariableop_resourceE
Amodel_4_batch_normalization_9_batchnorm_readvariableop_1_resourceE
Amodel_4_batch_normalization_9_batchnorm_readvariableop_2_resource3
/model_4_dense_12_matmul_readvariableop_resource4
0model_4_dense_12_biasadd_readvariableop_resourceD
@model_4_batch_normalization_10_batchnorm_readvariableop_resourceH
Dmodel_4_batch_normalization_10_batchnorm_mul_readvariableop_resourceF
Bmodel_4_batch_normalization_10_batchnorm_readvariableop_1_resourceF
Bmodel_4_batch_normalization_10_batchnorm_readvariableop_2_resource3
/model_4_dense_13_matmul_readvariableop_resource4
0model_4_dense_13_biasadd_readvariableop_resource3
/model_4_dense_14_matmul_readvariableop_resource4
0model_4_dense_14_biasadd_readvariableop_resource
identity��7model_4/batch_normalization_10/batchnorm/ReadVariableOp�9model_4/batch_normalization_10/batchnorm/ReadVariableOp_1�9model_4/batch_normalization_10/batchnorm/ReadVariableOp_2�;model_4/batch_normalization_10/batchnorm/mul/ReadVariableOp�6model_4/batch_normalization_9/batchnorm/ReadVariableOp�8model_4/batch_normalization_9/batchnorm/ReadVariableOp_1�8model_4/batch_normalization_9/batchnorm/ReadVariableOp_2�:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp�'model_4/dense_12/BiasAdd/ReadVariableOp�&model_4/dense_12/MatMul/ReadVariableOp�'model_4/dense_13/BiasAdd/ReadVariableOp�&model_4/dense_13/MatMul/ReadVariableOp�'model_4/dense_14/BiasAdd/ReadVariableOp�&model_4/dense_14/MatMul/ReadVariableOp�
*model_4/batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2,
*model_4/batch_normalization_9/LogicalAnd/x�
*model_4/batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2,
*model_4/batch_normalization_9/LogicalAnd/y�
(model_4/batch_normalization_9/LogicalAnd
LogicalAnd3model_4/batch_normalization_9/LogicalAnd/x:output:03model_4/batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2*
(model_4/batch_normalization_9/LogicalAnd�
6model_4/batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp?model_4_batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6model_4/batch_normalization_9/batchnorm/ReadVariableOp�
-model_4/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2/
-model_4/batch_normalization_9/batchnorm/add/y�
+model_4/batch_normalization_9/batchnorm/addAddV2>model_4/batch_normalization_9/batchnorm/ReadVariableOp:value:06model_4/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+model_4/batch_normalization_9/batchnorm/add�
-model_4/batch_normalization_9/batchnorm/RsqrtRsqrt/model_4/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-model_4/batch_normalization_9/batchnorm/Rsqrt�
:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_4_batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp�
+model_4/batch_normalization_9/batchnorm/mulMul1model_4/batch_normalization_9/batchnorm/Rsqrt:y:0Bmodel_4/batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+model_4/batch_normalization_9/batchnorm/mul�
-model_4/batch_normalization_9/batchnorm/mul_1Mulinput_8/model_4/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2/
-model_4/batch_normalization_9/batchnorm/mul_1�
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_4_batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_1�
-model_4/batch_normalization_9/batchnorm/mul_2Mul@model_4/batch_normalization_9/batchnorm/ReadVariableOp_1:value:0/model_4/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-model_4/batch_normalization_9/batchnorm/mul_2�
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_4_batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_2�
+model_4/batch_normalization_9/batchnorm/subSub@model_4/batch_normalization_9/batchnorm/ReadVariableOp_2:value:01model_4/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+model_4/batch_normalization_9/batchnorm/sub�
-model_4/batch_normalization_9/batchnorm/add_1AddV21model_4/batch_normalization_9/batchnorm/mul_1:z:0/model_4/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2/
-model_4/batch_normalization_9/batchnorm/add_1�
&model_4/dense_12/MatMul/ReadVariableOpReadVariableOp/model_4_dense_12_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02(
&model_4/dense_12/MatMul/ReadVariableOp�
model_4/dense_12/MatMulMatMul1model_4/batch_normalization_9/batchnorm/add_1:z:0.model_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
model_4/dense_12/MatMul�
'model_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_12_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02)
'model_4/dense_12/BiasAdd/ReadVariableOp�
model_4/dense_12/BiasAddBiasAdd!model_4/dense_12/MatMul:product:0/model_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
model_4/dense_12/BiasAdd�
+model_4/batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+model_4/batch_normalization_10/LogicalAnd/x�
+model_4/batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2-
+model_4/batch_normalization_10/LogicalAnd/y�
)model_4/batch_normalization_10/LogicalAnd
LogicalAnd4model_4/batch_normalization_10/LogicalAnd/x:output:04model_4/batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2+
)model_4/batch_normalization_10/LogicalAnd�
7model_4/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp@model_4_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype029
7model_4/batch_normalization_10/batchnorm/ReadVariableOp�
.model_4/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?20
.model_4/batch_normalization_10/batchnorm/add/y�
,model_4/batch_normalization_10/batchnorm/addAddV2?model_4/batch_normalization_10/batchnorm/ReadVariableOp:value:07model_4/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2.
,model_4/batch_normalization_10/batchnorm/add�
.model_4/batch_normalization_10/batchnorm/RsqrtRsqrt0model_4/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:(20
.model_4/batch_normalization_10/batchnorm/Rsqrt�
;model_4/batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_4_batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02=
;model_4/batch_normalization_10/batchnorm/mul/ReadVariableOp�
,model_4/batch_normalization_10/batchnorm/mulMul2model_4/batch_normalization_10/batchnorm/Rsqrt:y:0Cmodel_4/batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2.
,model_4/batch_normalization_10/batchnorm/mul�
.model_4/batch_normalization_10/batchnorm/mul_1Mul!model_4/dense_12/BiasAdd:output:00model_4/batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(20
.model_4/batch_normalization_10/batchnorm/mul_1�
9model_4/batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_4_batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype02;
9model_4/batch_normalization_10/batchnorm/ReadVariableOp_1�
.model_4/batch_normalization_10/batchnorm/mul_2MulAmodel_4/batch_normalization_10/batchnorm/ReadVariableOp_1:value:00model_4/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:(20
.model_4/batch_normalization_10/batchnorm/mul_2�
9model_4/batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_4_batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype02;
9model_4/batch_normalization_10/batchnorm/ReadVariableOp_2�
,model_4/batch_normalization_10/batchnorm/subSubAmodel_4/batch_normalization_10/batchnorm/ReadVariableOp_2:value:02model_4/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2.
,model_4/batch_normalization_10/batchnorm/sub�
.model_4/batch_normalization_10/batchnorm/add_1AddV22model_4/batch_normalization_10/batchnorm/mul_1:z:00model_4/batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(20
.model_4/batch_normalization_10/batchnorm/add_1�
!model_4/tf_op_layer_Relu_5/Relu_5Relu2model_4/batch_normalization_10/batchnorm/add_1:z:0*
T0*
_cloned(*'
_output_shapes
:���������(2#
!model_4/tf_op_layer_Relu_5/Relu_5�
!model_4/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_4/concatenate_3/concat/axis�
model_4/concatenate_3/concatConcatV2/model_4/tf_op_layer_Relu_5/Relu_5:activations:0input_9*model_4/concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������)2
model_4/concatenate_3/concat�
&model_4/dense_13/MatMul/ReadVariableOpReadVariableOp/model_4_dense_13_matmul_readvariableop_resource*
_output_shapes

:)*
dtype02(
&model_4/dense_13/MatMul/ReadVariableOp�
model_4/dense_13/MatMulMatMul%model_4/concatenate_3/concat:output:0.model_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_4/dense_13/MatMul�
'model_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_4/dense_13/BiasAdd/ReadVariableOp�
model_4/dense_13/BiasAddBiasAdd!model_4/dense_13/MatMul:product:0/model_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_4/dense_13/BiasAdd�
model_4/dense_13/ReluRelu!model_4/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model_4/dense_13/Relu�
&model_4/dense_14/MatMul/ReadVariableOpReadVariableOp/model_4_dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&model_4/dense_14/MatMul/ReadVariableOp�
model_4/dense_14/MatMulMatMul#model_4/dense_13/Relu:activations:0.model_4/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_4/dense_14/MatMul�
'model_4/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_4/dense_14/BiasAdd/ReadVariableOp�
model_4/dense_14/BiasAddBiasAdd!model_4/dense_14/MatMul:product:0/model_4/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_4/dense_14/BiasAdd�
model_4/dense_14/SigmoidSigmoid!model_4/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model_4/dense_14/Sigmoid�
IdentityIdentitymodel_4/dense_14/Sigmoid:y:08^model_4/batch_normalization_10/batchnorm/ReadVariableOp:^model_4/batch_normalization_10/batchnorm/ReadVariableOp_1:^model_4/batch_normalization_10/batchnorm/ReadVariableOp_2<^model_4/batch_normalization_10/batchnorm/mul/ReadVariableOp7^model_4/batch_normalization_9/batchnorm/ReadVariableOp9^model_4/batch_normalization_9/batchnorm/ReadVariableOp_19^model_4/batch_normalization_9/batchnorm/ReadVariableOp_2;^model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp(^model_4/dense_12/BiasAdd/ReadVariableOp'^model_4/dense_12/MatMul/ReadVariableOp(^model_4/dense_13/BiasAdd/ReadVariableOp'^model_4/dense_13/MatMul/ReadVariableOp(^model_4/dense_14/BiasAdd/ReadVariableOp'^model_4/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::2r
7model_4/batch_normalization_10/batchnorm/ReadVariableOp7model_4/batch_normalization_10/batchnorm/ReadVariableOp2v
9model_4/batch_normalization_10/batchnorm/ReadVariableOp_19model_4/batch_normalization_10/batchnorm/ReadVariableOp_12v
9model_4/batch_normalization_10/batchnorm/ReadVariableOp_29model_4/batch_normalization_10/batchnorm/ReadVariableOp_22z
;model_4/batch_normalization_10/batchnorm/mul/ReadVariableOp;model_4/batch_normalization_10/batchnorm/mul/ReadVariableOp2p
6model_4/batch_normalization_9/batchnorm/ReadVariableOp6model_4/batch_normalization_9/batchnorm/ReadVariableOp2t
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_18model_4/batch_normalization_9/batchnorm/ReadVariableOp_12t
8model_4/batch_normalization_9/batchnorm/ReadVariableOp_28model_4/batch_normalization_9/batchnorm/ReadVariableOp_22x
:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp:model_4/batch_normalization_9/batchnorm/mul/ReadVariableOp2R
'model_4/dense_12/BiasAdd/ReadVariableOp'model_4/dense_12/BiasAdd/ReadVariableOp2P
&model_4/dense_12/MatMul/ReadVariableOp&model_4/dense_12/MatMul/ReadVariableOp2R
'model_4/dense_13/BiasAdd/ReadVariableOp'model_4/dense_13/BiasAdd/ReadVariableOp2P
&model_4/dense_13/MatMul/ReadVariableOp&model_4/dense_13/MatMul/ReadVariableOp2R
'model_4/dense_14/BiasAdd/ReadVariableOp'model_4/dense_14/BiasAdd/ReadVariableOp2P
&model_4/dense_14/MatMul/ReadVariableOp&model_4/dense_14/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_8:'#
!
_user_specified_name	input_9
�2
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_476413

inputs
assignmovingavg_476386
assignmovingavg_1_476393)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/476386*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*)
_class
loc:@AssignMovingAvg/476386*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_476386*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/476386*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*)
_class
loc:@AssignMovingAvg/476386*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_476386AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/476386*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/476393*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*+
_class!
loc:@AssignMovingAvg_1/476393*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_476393*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/476393*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*+
_class!
loc:@AssignMovingAvg_1/476393*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_476393AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/476393*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
D__inference_dense_14_layer_call_and_return_conditional_losses_477482

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_476445

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_477275

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_9_layer_call_fn_477293

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_4764452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�,
�
__inference__traced_save_477562
file_prefix:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d5a3e9d4957842139b92bacb3ce1dc92/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*{
_input_shapesj
h: :::::(:(:(:(:(:(:):::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�*
�
C__inference_model_4_layer_call_and_return_conditional_losses_476805

inputs
inputs_18
4batch_normalization_9_statefulpartitionedcall_args_18
4batch_normalization_9_statefulpartitionedcall_args_28
4batch_normalization_9_statefulpartitionedcall_args_38
4batch_normalization_9_statefulpartitionedcall_args_4+
'dense_12_statefulpartitionedcall_args_1+
'dense_12_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_19
5batch_normalization_10_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_39
5batch_normalization_10_statefulpartitionedcall_args_4+
'dense_13_statefulpartitionedcall_args_1+
'dense_13_statefulpartitionedcall_args_2+
'dense_14_statefulpartitionedcall_args_1+
'dense_14_statefulpartitionedcall_args_2
identity��.batch_normalization_10/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCallinputs4batch_normalization_9_statefulpartitionedcall_args_14batch_normalization_9_statefulpartitionedcall_args_24batch_normalization_9_statefulpartitionedcall_args_34batch_normalization_9_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_4764132/
-batch_normalization_9/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0'dense_12_statefulpartitionedcall_args_1'dense_12_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4766382"
 dense_12/StatefulPartitionedCall�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:05batch_normalization_10_statefulpartitionedcall_args_15batch_normalization_10_statefulpartitionedcall_args_25batch_normalization_10_statefulpartitionedcall_args_35batch_normalization_10_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_47656120
.batch_normalization_10/StatefulPartitionedCall�
"tf_op_layer_Relu_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_4766782$
"tf_op_layer_Relu_5/PartitionedCall�
concatenate_3/PartitionedCallPartitionedCall+tf_op_layer_Relu_5/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������)*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4766932
concatenate_3/PartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0'dense_13_statefulpartitionedcall_args_1'dense_13_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4767132"
 dense_13/StatefulPartitionedCall�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0'dense_14_statefulpartitionedcall_args_1'dense_14_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_4767362"
 dense_14/StatefulPartitionedCall�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
D__inference_dense_12_layer_call_and_return_conditional_losses_477303

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_9_layer_call_fn_477284

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_4764132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�G
�	
"__inference__traced_restore_477622
file_prefix0
,assignvariableop_batch_normalization_9_gamma1
-assignvariableop_1_batch_normalization_9_beta8
4assignvariableop_2_batch_normalization_9_moving_mean<
8assignvariableop_3_batch_normalization_9_moving_variance&
"assignvariableop_4_dense_12_kernel$
 assignvariableop_5_dense_12_bias3
/assignvariableop_6_batch_normalization_10_gamma2
.assignvariableop_7_batch_normalization_10_beta9
5assignvariableop_8_batch_normalization_10_moving_mean=
9assignvariableop_9_batch_normalization_10_moving_variance'
#assignvariableop_10_dense_13_kernel%
!assignvariableop_11_dense_13_bias'
#assignvariableop_12_dense_14_kernel%
!assignvariableop_13_dense_14_bias
assignvariableop_14_total
assignvariableop_15_count
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_9_gammaIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_9_betaIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_9_moving_meanIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_9_moving_varianceIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_12_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_12_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_10_gammaIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_10_betaIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_10_moving_meanIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_10_moving_varianceIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_13_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_13_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_14_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_14_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16�
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_477412

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������(::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_477447
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������)2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������)2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������(:���������:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_476593

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������(::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_dense_14_layer_call_fn_477489

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_4767362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
l
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_477435
inputs_0
identityc
Relu_5Reluinputs_0*
T0*
_cloned(*'
_output_shapes
:���������(2
Relu_5h
IdentityIdentityRelu_5:activations:0*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������(:( $
"
_user_specified_name
inputs/0
�
�
$__inference_signature_wrapper_476965
input_8
input_9"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_8input_9statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_4763042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_8:'#
!
_user_specified_name	input_9
�a
�
C__inference_model_4_layer_call_and_return_conditional_losses_477133
inputs_0
inputs_1;
7batch_normalization_9_batchnorm_readvariableop_resource?
;batch_normalization_9_batchnorm_mul_readvariableop_resource=
9batch_normalization_9_batchnorm_readvariableop_1_resource=
9batch_normalization_9_batchnorm_readvariableop_2_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource<
8batch_normalization_10_batchnorm_readvariableop_resource@
<batch_normalization_10_batchnorm_mul_readvariableop_resource>
:batch_normalization_10_batchnorm_readvariableop_1_resource>
:batch_normalization_10_batchnorm_readvariableop_2_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource
identity��/batch_normalization_10/batchnorm/ReadVariableOp�1batch_normalization_10/batchnorm/ReadVariableOp_1�1batch_normalization_10/batchnorm/ReadVariableOp_2�3batch_normalization_10/batchnorm/mul/ReadVariableOp�.batch_normalization_9/batchnorm/ReadVariableOp�0batch_normalization_9/batchnorm/ReadVariableOp_1�0batch_normalization_9/batchnorm/ReadVariableOp_2�2batch_normalization_9/batchnorm/mul/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
.batch_normalization_9/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_9_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_9/batchnorm/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV26batch_normalization_9/batchnorm/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt�
2batch_normalization_9/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_9_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_9/batchnorm/mul/ReadVariableOp�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:0:batch_normalization_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Mulinputs_0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/mul_1�
0batch_normalization_9/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_9/batchnorm/ReadVariableOp_1�
%batch_normalization_9/batchnorm/mul_2Mul8batch_normalization_9/batchnorm/ReadVariableOp_1:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2�
0batch_normalization_9/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_9_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_9/batchnorm/ReadVariableOp_2�
#batch_normalization_9/batchnorm/subSub8batch_normalization_9/batchnorm/ReadVariableOp_2:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/add_1�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMul)batch_normalization_9/batchnorm/add_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_12/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype021
/batch_normalization_10/batchnorm/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV27batch_normalization_10/batchnorm/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_10/batchnorm/Rsqrt�
3batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype025
3batch_normalization_10/batchnorm/mul/ReadVariableOp�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:0;batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_12/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_10/batchnorm/mul_1�
1batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_1�
&batch_normalization_10/batchnorm/mul_2Mul9batch_normalization_10/batchnorm/ReadVariableOp_1:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_10/batchnorm/mul_2�
1batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype023
1batch_normalization_10/batchnorm/ReadVariableOp_2�
$batch_normalization_10/batchnorm/subSub9batch_normalization_10/batchnorm/ReadVariableOp_2:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_10/batchnorm/add_1�
tf_op_layer_Relu_5/Relu_5Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*
_cloned(*'
_output_shapes
:���������(2
tf_op_layer_Relu_5/Relu_5x
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis�
concatenate_3/concatConcatV2'tf_op_layer_Relu_5/Relu_5:activations:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������)2
concatenate_3/concat�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:)*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulconcatenate_3/concat:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_13/Relu�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_14/BiasAdd|
dense_14/SigmoidSigmoiddense_14/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_14/Sigmoid�
IdentityIdentitydense_14/Sigmoid:y:00^batch_normalization_10/batchnorm/ReadVariableOp2^batch_normalization_10/batchnorm/ReadVariableOp_12^batch_normalization_10/batchnorm/ReadVariableOp_24^batch_normalization_10/batchnorm/mul/ReadVariableOp/^batch_normalization_9/batchnorm/ReadVariableOp1^batch_normalization_9/batchnorm/ReadVariableOp_11^batch_normalization_9/batchnorm/ReadVariableOp_23^batch_normalization_9/batchnorm/mul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::2b
/batch_normalization_10/batchnorm/ReadVariableOp/batch_normalization_10/batchnorm/ReadVariableOp2f
1batch_normalization_10/batchnorm/ReadVariableOp_11batch_normalization_10/batchnorm/ReadVariableOp_12f
1batch_normalization_10/batchnorm/ReadVariableOp_21batch_normalization_10/batchnorm/ReadVariableOp_22j
3batch_normalization_10/batchnorm/mul/ReadVariableOp3batch_normalization_10/batchnorm/mul/ReadVariableOp2`
.batch_normalization_9/batchnorm/ReadVariableOp.batch_normalization_9/batchnorm/ReadVariableOp2d
0batch_normalization_9/batchnorm/ReadVariableOp_10batch_normalization_9/batchnorm/ReadVariableOp_12d
0batch_normalization_9/batchnorm/ReadVariableOp_20batch_normalization_9/batchnorm/ReadVariableOp_22h
2batch_normalization_9/batchnorm/mul/ReadVariableOp2batch_normalization_9/batchnorm/mul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�2
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_476561

inputs
assignmovingavg_476534
assignmovingavg_1_476541)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:(2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������(2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/476534*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*)
_class
loc:@AssignMovingAvg/476534*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_476534*
_output_shapes
:(*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/476534*
_output_shapes
:(2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*)
_class
loc:@AssignMovingAvg/476534*
_output_shapes
:(2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_476534AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/476534*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/476541*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*+
_class!
loc:@AssignMovingAvg_1/476541*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_476541*
_output_shapes
:(*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/476541*
_output_shapes
:(2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*+
_class!
loc:@AssignMovingAvg_1/476541*
_output_shapes
:(2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_476541AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/476541*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:���������(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������(::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
D__inference_dense_13_layer_call_and_return_conditional_losses_476713

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:)*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������)::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_model_4_layer_call_fn_477153
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_4768052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�*
�
C__inference_model_4_layer_call_and_return_conditional_losses_476749
input_8
input_98
4batch_normalization_9_statefulpartitionedcall_args_18
4batch_normalization_9_statefulpartitionedcall_args_28
4batch_normalization_9_statefulpartitionedcall_args_38
4batch_normalization_9_statefulpartitionedcall_args_4+
'dense_12_statefulpartitionedcall_args_1+
'dense_12_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_19
5batch_normalization_10_statefulpartitionedcall_args_29
5batch_normalization_10_statefulpartitionedcall_args_39
5batch_normalization_10_statefulpartitionedcall_args_4+
'dense_13_statefulpartitionedcall_args_1+
'dense_13_statefulpartitionedcall_args_2+
'dense_14_statefulpartitionedcall_args_1+
'dense_14_statefulpartitionedcall_args_2
identity��.batch_normalization_10/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCallinput_84batch_normalization_9_statefulpartitionedcall_args_14batch_normalization_9_statefulpartitionedcall_args_24batch_normalization_9_statefulpartitionedcall_args_34batch_normalization_9_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_4764132/
-batch_normalization_9/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0'dense_12_statefulpartitionedcall_args_1'dense_12_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_4766382"
 dense_12/StatefulPartitionedCall�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:05batch_normalization_10_statefulpartitionedcall_args_15batch_normalization_10_statefulpartitionedcall_args_25batch_normalization_10_statefulpartitionedcall_args_35batch_normalization_10_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_47656120
.batch_normalization_10/StatefulPartitionedCall�
"tf_op_layer_Relu_5/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������(*-
config_proto

CPU

GPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_4766782$
"tf_op_layer_Relu_5/PartitionedCall�
concatenate_3/PartitionedCallPartitionedCall+tf_op_layer_Relu_5/PartitionedCall:output:0input_9*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������)*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4766932
concatenate_3/PartitionedCall�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0'dense_13_statefulpartitionedcall_args_1'dense_13_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_4767132"
 dense_13/StatefulPartitionedCall�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0'dense_14_statefulpartitionedcall_args_1'dense_14_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_4767362"
 dense_14/StatefulPartitionedCall�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:���������:���������::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:' #
!
_user_specified_name	input_8:'#
!
_user_specified_name	input_9"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_80
serving_default_input_8:0���������
;
input_90
serving_default_input_9:0���������<
dense_140
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�J
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
i_default_save_signature
*j&call_and_return_all_conditional_losses
k__call__"�F
_tf_keras_model�F{"class_name": "Model", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float64", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_5", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_5", "op": "Relu", "input": ["batch_normalization_10/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}, "name": "tf_op_layer_Relu_5", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float64", "axis": 1}, "name": "concatenate_3", "inbound_nodes": [[["tf_op_layer_Relu_5", 0, 0, {}], ["input_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float64", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.15811388300841897, "maxval": 0.15811388300841897, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.15811388300841897, "maxval": 0.15811388300841897, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float64", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.001, "maxval": 0.001, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.001, "maxval": 0.001, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}], "input_layers": [["input_8", 0, 0], ["input_9", 0, 0]], "output_layers": [["dense_14", 0, 0]]}, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_9", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float64", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_10", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_5", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_5", "op": "Relu", "input": ["batch_normalization_10/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}, "name": "tf_op_layer_Relu_5", "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float64", "axis": 1}, "name": "concatenate_3", "inbound_nodes": [[["tf_op_layer_Relu_5", 0, 0, {}], ["input_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float64", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.15811388300841897, "maxval": 0.15811388300841897, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.15811388300841897, "maxval": 0.15811388300841897, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float64", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.001, "maxval": 0.001, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.001, "maxval": 0.001, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}], "input_layers": [["input_8", 0, 0], ["input_9", 0, 0]], "output_layers": [["dense_14", 0, 0]]}}, "training_config": {"loss": "MSE", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_8", "dtype": "float64", "sparse": false, "ragged": false, "batch_input_shape": [null, 2], "config": {"batch_input_shape": [null, 2], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_8"}}
�
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 2}}}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "dense_12", "trainable": true, "dtype": "float64", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
�
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
*p&call_and_return_all_conditional_losses
q__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 40}}}}
�
(	constants
)regularization_losses
*trainable_variables
+	variables
,	keras_api
*r&call_and_return_all_conditional_losses
s__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Relu_5", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "Relu_5", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_5", "op": "Relu", "input": ["batch_normalization_10/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_9", "dtype": "float64", "sparse": false, "ragged": false, "batch_input_shape": [null, 1], "config": {"batch_input_shape": [null, 1], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_9"}}
�
-regularization_losses
.trainable_variables
/	variables
0	keras_api
*t&call_and_return_all_conditional_losses
u__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float64", "axis": 1}}
�

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
*v&call_and_return_all_conditional_losses
w__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "dense_13", "trainable": true, "dtype": "float64", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.15811388300841897, "maxval": 0.15811388300841897, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.15811388300841897, "maxval": 0.15811388300841897, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 41}}}}
�

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
*x&call_and_return_all_conditional_losses
y__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "dense_14", "trainable": true, "dtype": "float64", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.001, "maxval": 0.001, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.001, "maxval": 0.001, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}}
"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
 4
!5
16
27
78
89"
trackable_list_wrapper
�
0
1
2
3
4
5
 6
!7
"8
#9
110
211
712
813"
trackable_list_wrapper
�
regularization_losses
trainable_variables

=layers
>metrics
?layer_regularization_losses
@non_trainable_variables
	variables
k__call__
i_default_save_signature
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
,
zserving_default"
signature_map
 "
trackable_list_wrapper
):'2batch_normalization_9/gamma
(:&2batch_normalization_9/beta
1:/ (2!batch_normalization_9/moving_mean
5:3 (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
regularization_losses
trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables
	variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
!:(2dense_12/kernel
:(2dense_12/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
	variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:((2batch_normalization_10/gamma
):'(2batch_normalization_10/beta
2:0( (2"batch_normalization_10/moving_mean
6:4( (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
�
$regularization_losses
%trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Lnon_trainable_variables
&	variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
)regularization_losses
*trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Pnon_trainable_variables
+	variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
-regularization_losses
.trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tnon_trainable_variables
/	variables
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:)2dense_13/kernel
:2dense_13/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
3regularization_losses
4trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xnon_trainable_variables
5	variables
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
!:2dense_14/kernel
:2dense_14/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�
9regularization_losses
:trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\non_trainable_variables
;	variables
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
'
]0"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
"2
#3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
.
"0
#1"
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
	^total
	_count
`
_fn_kwargs
aregularization_losses
btrainable_variables
c	variables
d	keras_api
*{&call_and_return_all_conditional_losses
|__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float64"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
�
aregularization_losses
btrainable_variables

elayers
fmetrics
glayer_regularization_losses
hnon_trainable_variables
c	variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
�2�
!__inference__wrapped_model_476304�
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
annotations� *N�K
I�F
!�
input_8���������
!�
input_9���������
�2�
C__inference_model_4_layer_call_and_return_conditional_losses_477067
C__inference_model_4_layer_call_and_return_conditional_losses_477133
C__inference_model_4_layer_call_and_return_conditional_losses_476749
C__inference_model_4_layer_call_and_return_conditional_losses_476775�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_model_4_layer_call_fn_476868
(__inference_model_4_layer_call_fn_476822
(__inference_model_4_layer_call_fn_477173
(__inference_model_4_layer_call_fn_477153�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_477252
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_477275�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_9_layer_call_fn_477293
6__inference_batch_normalization_9_layer_call_fn_477284�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dense_12_layer_call_and_return_conditional_losses_477303�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
)__inference_dense_12_layer_call_fn_477310�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_477412
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_477389�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_10_layer_call_fn_477430
7__inference_batch_normalization_10_layer_call_fn_477421�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_477435�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
3__inference_tf_op_layer_Relu_5_layer_call_fn_477440�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
I__inference_concatenate_3_layer_call_and_return_conditional_losses_477447�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
.__inference_concatenate_3_layer_call_fn_477453�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
D__inference_dense_13_layer_call_and_return_conditional_losses_477464�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
)__inference_dense_13_layer_call_fn_477471�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
D__inference_dense_14_layer_call_and_return_conditional_losses_477482�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
)__inference_dense_14_layer_call_fn_477489�
���
FullArgSpec
args�
jself
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
annotations� *
 
:B8
$__inference_signature_wrapper_476965input_8input_9
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
!__inference__wrapped_model_476304�# "!1278X�U
N�K
I�F
!�
input_8���������
!�
input_9���������
� "3�0
.
dense_14"�
dense_14����������
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_477389b"# !3�0
)�&
 �
inputs���������(
p
� "%�"
�
0���������(
� �
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_477412b# "!3�0
)�&
 �
inputs���������(
p 
� "%�"
�
0���������(
� �
7__inference_batch_normalization_10_layer_call_fn_477421U"# !3�0
)�&
 �
inputs���������(
p
� "����������(�
7__inference_batch_normalization_10_layer_call_fn_477430U# "!3�0
)�&
 �
inputs���������(
p 
� "����������(�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_477252b3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_477275b3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
6__inference_batch_normalization_9_layer_call_fn_477284U3�0
)�&
 �
inputs���������
p
� "�����������
6__inference_batch_normalization_9_layer_call_fn_477293U3�0
)�&
 �
inputs���������
p 
� "�����������
I__inference_concatenate_3_layer_call_and_return_conditional_losses_477447�Z�W
P�M
K�H
"�
inputs/0���������(
"�
inputs/1���������
� "%�"
�
0���������)
� �
.__inference_concatenate_3_layer_call_fn_477453vZ�W
P�M
K�H
"�
inputs/0���������(
"�
inputs/1���������
� "����������)�
D__inference_dense_12_layer_call_and_return_conditional_losses_477303\/�,
%�"
 �
inputs���������
� "%�"
�
0���������(
� |
)__inference_dense_12_layer_call_fn_477310O/�,
%�"
 �
inputs���������
� "����������(�
D__inference_dense_13_layer_call_and_return_conditional_losses_477464\12/�,
%�"
 �
inputs���������)
� "%�"
�
0���������
� |
)__inference_dense_13_layer_call_fn_477471O12/�,
%�"
 �
inputs���������)
� "�����������
D__inference_dense_14_layer_call_and_return_conditional_losses_477482\78/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
)__inference_dense_14_layer_call_fn_477489O78/�,
%�"
 �
inputs���������
� "�����������
C__inference_model_4_layer_call_and_return_conditional_losses_476749�"# !1278`�]
V�S
I�F
!�
input_8���������
!�
input_9���������
p

 
� "%�"
�
0���������
� �
C__inference_model_4_layer_call_and_return_conditional_losses_476775�# "!1278`�]
V�S
I�F
!�
input_8���������
!�
input_9���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_4_layer_call_and_return_conditional_losses_477067�"# !1278b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "%�"
�
0���������
� �
C__inference_model_4_layer_call_and_return_conditional_losses_477133�# "!1278b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "%�"
�
0���������
� �
(__inference_model_4_layer_call_fn_476822�"# !1278`�]
V�S
I�F
!�
input_8���������
!�
input_9���������
p

 
� "�����������
(__inference_model_4_layer_call_fn_476868�# "!1278`�]
V�S
I�F
!�
input_8���������
!�
input_9���������
p 

 
� "�����������
(__inference_model_4_layer_call_fn_477153�"# !1278b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "�����������
(__inference_model_4_layer_call_fn_477173�# "!1278b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "�����������
$__inference_signature_wrapper_476965�# "!1278i�f
� 
_�\
,
input_8!�
input_8���������
,
input_9!�
input_9���������"3�0
.
dense_14"�
dense_14����������
N__inference_tf_op_layer_Relu_5_layer_call_and_return_conditional_losses_477435_6�3
,�)
'�$
"�
inputs/0���������(
� "%�"
�
0���������(
� �
3__inference_tf_op_layer_Relu_5_layer_call_fn_477440R6�3
,�)
'�$
"�
inputs/0���������(
� "����������(