��
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
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_15/gamma
�
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_15/beta
�
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:*
dtype0
�
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_15/moving_mean
�
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:*
dtype0
�
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_15/moving_variance
�
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:*
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	�*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_16/gamma
�
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_16/beta
�
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes	
:�*
dtype0
|
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_19/kernel
u
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel* 
_output_shapes
:
��*
dtype0
s
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_19/bias
l
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_17/gamma
�
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_17/beta
�
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes	
:�*
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	�*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�,B�, B�,
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%	variables
&trainable_variables
'regularization_losses
(	keras_api
a
)	constants
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
�
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:trainable_variables
;regularization_losses
<	keras_api
a
=	constants
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
a
H	constants
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
 
�
0
1
2
3
4
5
!6
"7
#8
$9
.10
/11
512
613
714
815
B16
C17
V
0
1
2
3
!4
"5
.6
/7
58
69
B10
C11
 
�

Mlayers
	variables
Nlayer_regularization_losses
Ometrics
trainable_variables
Pnon_trainable_variables
regularization_losses
 
 
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
�

Qlayers
	variables
Rlayer_regularization_losses
Smetrics
trainable_variables
Tnon_trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�

Ulayers
	variables
Vlayer_regularization_losses
Wmetrics
trainable_variables
Xnon_trainable_variables
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
#2
$3

!0
"1
 
�

Ylayers
%	variables
Zlayer_regularization_losses
[metrics
&trainable_variables
\non_trainable_variables
'regularization_losses
 
 
 
 
�

]layers
*	variables
^layer_regularization_losses
_metrics
+trainable_variables
`non_trainable_variables
,regularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
�

alayers
0	variables
blayer_regularization_losses
cmetrics
1trainable_variables
dnon_trainable_variables
2regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

50
61
72
83

50
61
 
�

elayers
9	variables
flayer_regularization_losses
gmetrics
:trainable_variables
hnon_trainable_variables
;regularization_losses
 
 
 
 
�

ilayers
>	variables
jlayer_regularization_losses
kmetrics
?trainable_variables
lnon_trainable_variables
@regularization_losses
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
�

mlayers
D	variables
nlayer_regularization_losses
ometrics
Etrainable_variables
pnon_trainable_variables
Fregularization_losses
 
 
 
 
�

qlayers
I	variables
rlayer_regularization_losses
smetrics
Jtrainable_variables
tnon_trainable_variables
Kregularization_losses
F
0
1
2
3
4
5
6
7
	8

9
 
 
*
0
1
#2
$3
74
85
 
 
 

0
1
 
 
 
 
 
 
 

#0
$1
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

70
81
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
{
serving_default_input_10Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10&batch_normalization_15/moving_variancebatch_normalization_15/gamma"batch_normalization_15/moving_meanbatch_normalization_15/betadense_18/kerneldense_18/bias&batch_normalization_16/moving_variancebatch_normalization_16/gamma"batch_normalization_16/moving_meanbatch_normalization_16/betadense_19/kerneldense_19/bias&batch_normalization_17/moving_variancebatch_normalization_17/gamma"batch_normalization_17/moving_meanbatch_normalization_17/betadense_20/kerneldense_20/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*0
f+R)
'__inference_signature_wrapper_203566876
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpConst*
Tin
2*
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
"__inference__traced_save_203567665
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variancedense_18/kerneldense_18/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variancedense_19/kerneldense_19/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variancedense_20/kerneldense_20/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*.
f)R'
%__inference__traced_restore_203567731��
�
�
G__inference_dense_19_layer_call_and_return_conditional_losses_203567421

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
:__inference_batch_normalization_15_layer_call_fn_203567255

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
GPU2*0J 8*^
fYRW
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2035660822
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
�
�
G__inference_dense_19_layer_call_and_return_conditional_losses_203566512

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�0
�	
"__inference__traced_save_203567665
file_prefix;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1dbc59a28144470890c493ff4bf9a0eb/part2
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
ShardedFilename�	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 * 
dtypes
22
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::	�:�:�:�:�:�:
��:�:�:�:�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203566262

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
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
��
�
F__inference_model_6_layer_call_and_return_conditional_losses_203567014

inputs4
0batch_normalization_15_assignmovingavg_2035668906
2batch_normalization_15_assignmovingavg_1_203566897@
<batch_normalization_15_batchnorm_mul_readvariableop_resource<
8batch_normalization_15_batchnorm_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource4
0batch_normalization_16_assignmovingavg_2035669336
2batch_normalization_16_assignmovingavg_1_203566940@
<batch_normalization_16_batchnorm_mul_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource4
0batch_normalization_17_assignmovingavg_2035669776
2batch_normalization_17_assignmovingavg_1_203566984@
<batch_normalization_17_batchnorm_mul_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity��:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_15/AssignMovingAvg/ReadVariableOp�<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_15/batchnorm/ReadVariableOp�3batch_normalization_15/batchnorm/mul/ReadVariableOp�:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_16/batchnorm/ReadVariableOp�3batch_normalization_16/batchnorm/mul/ReadVariableOp�:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_17/batchnorm/ReadVariableOp�3batch_normalization_17/batchnorm/mul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�
#batch_normalization_15/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_15/LogicalAnd/x�
#batch_normalization_15/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_15/LogicalAnd/y�
!batch_normalization_15/LogicalAnd
LogicalAnd,batch_normalization_15/LogicalAnd/x:output:0,batch_normalization_15/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_15/LogicalAnd�
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_15/moments/mean/reduction_indices�
#batch_normalization_15/moments/meanMeaninputs>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_15/moments/mean�
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_15/moments/StopGradient�
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferenceinputs4batch_normalization_15/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_15/moments/SquaredDifference�
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_15/moments/variance/reduction_indices�
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_15/moments/variance�
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_15/moments/Squeeze�
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_15/moments/Squeeze_1�
,batch_normalization_15/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_15/AssignMovingAvg/203566890*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_15/AssignMovingAvg/decay�
+batch_normalization_15/AssignMovingAvg/CastCast5batch_normalization_15/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_15/AssignMovingAvg/203566890*
_output_shapes
: 2-
+batch_normalization_15/AssignMovingAvg/Cast�
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_15_assignmovingavg_203566890*
_output_shapes
:*
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOp�
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_15/AssignMovingAvg/203566890*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/sub�
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:0/batch_normalization_15/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_15/AssignMovingAvg/203566890*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/mul�
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_15_assignmovingavg_203566890.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_15/AssignMovingAvg/203566890*
_output_shapes
 *
dtype02<
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_15/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_15/AssignMovingAvg_1/203566897*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_15/AssignMovingAvg_1/decay�
-batch_normalization_15/AssignMovingAvg_1/CastCast7batch_normalization_15/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_15/AssignMovingAvg_1/203566897*
_output_shapes
: 2/
-batch_normalization_15/AssignMovingAvg_1/Cast�
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_15_assignmovingavg_1_203566897*
_output_shapes
:*
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_15/AssignMovingAvg_1/203566897*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/sub�
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:01batch_normalization_15/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_15/AssignMovingAvg_1/203566897*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/mul�
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_15_assignmovingavg_1_2035668970batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_15/AssignMovingAvg_1/203566897*
_output_shapes
 *
dtype02>
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp�
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_15/batchnorm/add/y�
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt�
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOp�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul�
&batch_normalization_15/batchnorm/mul_1Mulinputs(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/mul_1�
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2�
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOp�
$batch_normalization_15/batchnorm/subSub7batch_normalization_15/batchnorm/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/add_1�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMul*batch_normalization_15/batchnorm/add_1:z:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/BiasAdd�
#batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/x�
#batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/y�
!batch_normalization_16/LogicalAnd
LogicalAnd,batch_normalization_16/LogicalAnd/x:output:0,batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_16/LogicalAnd�
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_16/moments/mean/reduction_indices�
#batch_normalization_16/moments/meanMeandense_18/BiasAdd:output:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_16/moments/mean�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_16/moments/StopGradient�
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_18/BiasAdd:output:04batch_normalization_16/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_16/moments/SquaredDifference�
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_16/moments/variance/reduction_indices�
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_16/moments/variance�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_16/moments/Squeeze�
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_16/moments/Squeeze_1�
,batch_normalization_16/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_16/AssignMovingAvg/203566933*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_16/AssignMovingAvg/decay�
+batch_normalization_16/AssignMovingAvg/CastCast5batch_normalization_16/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_16/AssignMovingAvg/203566933*
_output_shapes
: 2-
+batch_normalization_16/AssignMovingAvg/Cast�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_16_assignmovingavg_203566933*
_output_shapes	
:�*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_16/AssignMovingAvg/203566933*
_output_shapes	
:�2,
*batch_normalization_16/AssignMovingAvg/sub�
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:0/batch_normalization_16/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_16/AssignMovingAvg/203566933*
_output_shapes	
:�2,
*batch_normalization_16/AssignMovingAvg/mul�
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_16_assignmovingavg_203566933.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_16/AssignMovingAvg/203566933*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_16/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_16/AssignMovingAvg_1/203566940*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_16/AssignMovingAvg_1/decay�
-batch_normalization_16/AssignMovingAvg_1/CastCast7batch_normalization_16/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_16/AssignMovingAvg_1/203566940*
_output_shapes
: 2/
-batch_normalization_16/AssignMovingAvg_1/Cast�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_16_assignmovingavg_1_203566940*
_output_shapes	
:�*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_16/AssignMovingAvg_1/203566940*
_output_shapes	
:�2.
,batch_normalization_16/AssignMovingAvg_1/sub�
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:01batch_normalization_16/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_16/AssignMovingAvg_1/203566940*
_output_shapes	
:�2.
,batch_normalization_16/AssignMovingAvg_1/mul�
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_16_assignmovingavg_1_2035669400batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_16/AssignMovingAvg_1/203566940*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp�
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_16/batchnorm/add/y�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_16/batchnorm/add�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_16/batchnorm/Rsqrt�
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOp�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_16/batchnorm/mul�
&batch_normalization_16/batchnorm/mul_1Muldense_18/BiasAdd:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_16/batchnorm/mul_1�
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_16/batchnorm/mul_2�
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOp�
$batch_normalization_16/batchnorm/subSub7batch_normalization_16/batchnorm/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_16/batchnorm/sub�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_16/batchnorm/add_1�
tf_op_layer_Relu_9/Relu_9Relu*batch_normalization_16/batchnorm/add_1:z:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Relu_9/Relu_9�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMul'tf_op_layer_Relu_9/Relu_9:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/BiasAdd�
#batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/x�
#batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/y�
!batch_normalization_17/LogicalAnd
LogicalAnd,batch_normalization_17/LogicalAnd/x:output:0,batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_17/LogicalAnd�
5batch_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_17/moments/mean/reduction_indices�
#batch_normalization_17/moments/meanMeandense_19/BiasAdd:output:0>batch_normalization_17/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_17/moments/mean�
+batch_normalization_17/moments/StopGradientStopGradient,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_17/moments/StopGradient�
0batch_normalization_17/moments/SquaredDifferenceSquaredDifferencedense_19/BiasAdd:output:04batch_normalization_17/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_17/moments/SquaredDifference�
9batch_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_17/moments/variance/reduction_indices�
'batch_normalization_17/moments/varianceMean4batch_normalization_17/moments/SquaredDifference:z:0Bbatch_normalization_17/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_17/moments/variance�
&batch_normalization_17/moments/SqueezeSqueeze,batch_normalization_17/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_17/moments/Squeeze�
(batch_normalization_17/moments/Squeeze_1Squeeze0batch_normalization_17/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_17/moments/Squeeze_1�
,batch_normalization_17/AssignMovingAvg/decayConst*C
_class9
75loc:@batch_normalization_17/AssignMovingAvg/203566977*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_17/AssignMovingAvg/decay�
+batch_normalization_17/AssignMovingAvg/CastCast5batch_normalization_17/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*C
_class9
75loc:@batch_normalization_17/AssignMovingAvg/203566977*
_output_shapes
: 2-
+batch_normalization_17/AssignMovingAvg/Cast�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_17_assignmovingavg_203566977*
_output_shapes	
:�*
dtype027
5batch_normalization_17/AssignMovingAvg/ReadVariableOp�
*batch_normalization_17/AssignMovingAvg/subSub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_17/moments/Squeeze:output:0*
T0*C
_class9
75loc:@batch_normalization_17/AssignMovingAvg/203566977*
_output_shapes	
:�2,
*batch_normalization_17/AssignMovingAvg/sub�
*batch_normalization_17/AssignMovingAvg/mulMul.batch_normalization_17/AssignMovingAvg/sub:z:0/batch_normalization_17/AssignMovingAvg/Cast:y:0*
T0*C
_class9
75loc:@batch_normalization_17/AssignMovingAvg/203566977*
_output_shapes	
:�2,
*batch_normalization_17/AssignMovingAvg/mul�
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_17_assignmovingavg_203566977.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*C
_class9
75loc:@batch_normalization_17/AssignMovingAvg/203566977*
_output_shapes
 *
dtype02<
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_17/AssignMovingAvg_1/decayConst*E
_class;
97loc:@batch_normalization_17/AssignMovingAvg_1/203566984*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_17/AssignMovingAvg_1/decay�
-batch_normalization_17/AssignMovingAvg_1/CastCast7batch_normalization_17/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*E
_class;
97loc:@batch_normalization_17/AssignMovingAvg_1/203566984*
_output_shapes
: 2/
-batch_normalization_17/AssignMovingAvg_1/Cast�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_17_assignmovingavg_1_203566984*
_output_shapes	
:�*
dtype029
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_17/AssignMovingAvg_1/subSub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_17/moments/Squeeze_1:output:0*
T0*E
_class;
97loc:@batch_normalization_17/AssignMovingAvg_1/203566984*
_output_shapes	
:�2.
,batch_normalization_17/AssignMovingAvg_1/sub�
,batch_normalization_17/AssignMovingAvg_1/mulMul0batch_normalization_17/AssignMovingAvg_1/sub:z:01batch_normalization_17/AssignMovingAvg_1/Cast:y:0*
T0*E
_class;
97loc:@batch_normalization_17/AssignMovingAvg_1/203566984*
_output_shapes	
:�2.
,batch_normalization_17/AssignMovingAvg_1/mul�
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_17_assignmovingavg_1_2035669840batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*E
_class;
97loc:@batch_normalization_17/AssignMovingAvg_1/203566984*
_output_shapes
 *
dtype02>
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp�
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_17/batchnorm/add/y�
$batch_normalization_17/batchnorm/addAddV21batch_normalization_17/moments/Squeeze_1:output:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_17/batchnorm/add�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_17/batchnorm/Rsqrt�
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOp�
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_17/batchnorm/mul�
&batch_normalization_17/batchnorm/mul_1Muldense_19/BiasAdd:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_17/batchnorm/mul_1�
&batch_normalization_17/batchnorm/mul_2Mul/batch_normalization_17/moments/Squeeze:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_17/batchnorm/mul_2�
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOp�
$batch_normalization_17/batchnorm/subSub7batch_normalization_17/batchnorm/ReadVariableOp:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_17/batchnorm/sub�
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_17/batchnorm/add_1�
tf_op_layer_Relu_10/Relu_10Relu*batch_normalization_17/batchnorm/add_1:z:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Relu_10/Relu_10�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)tf_op_layer_Relu_10/Relu_10:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdds
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Tanh
tf_op_layer_Mul_3/Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB 2      �?2
tf_op_layer_Mul_3/Mul_3/y�
tf_op_layer_Mul_3/Mul_3Muldense_20/Tanh:y:0"tf_op_layer_Mul_3/Mul_3/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������2
tf_op_layer_Mul_3/Mul_3�

IdentityIdentitytf_op_layer_Mul_3/Mul_3:z:0;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_15/AssignMovingAvg/ReadVariableOp=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_15/batchnorm/ReadVariableOp4^batch_normalization_15/batchnorm/mul/ReadVariableOp;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_16/AssignMovingAvg/ReadVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_16/batchnorm/ReadVariableOp4^batch_normalization_16/batchnorm/mul/ReadVariableOp;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_17/AssignMovingAvg/ReadVariableOp=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_17/batchnorm/ReadVariableOp4^batch_normalization_17/batchnorm/mul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::2x
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_15/batchnorm/ReadVariableOp/batch_normalization_15/batchnorm/ReadVariableOp2j
3batch_normalization_15/batchnorm/mul/ReadVariableOp3batch_normalization_15/batchnorm/mul/ReadVariableOp2x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_16/batchnorm/ReadVariableOp/batch_normalization_16/batchnorm/ReadVariableOp2j
3batch_normalization_16/batchnorm/mul/ReadVariableOp3batch_normalization_16/batchnorm/mul/ReadVariableOp2x
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_17/batchnorm/ReadVariableOp/batch_normalization_17/batchnorm/ReadVariableOp2j
3batch_normalization_17/batchnorm/mul/ReadVariableOp3batch_normalization_17/batchnorm/mul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
T
6__inference_tf_op_layer_Relu_9_layer_call_fn_203567411
inputs_0
identity�
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_2035664942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:( $
"
_user_specified_name
inputs/0
�
n
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_203566552

inputs
identityd
Relu_10Reluinputs*
T0*
_cloned(*(
_output_shapes
:����������2	
Relu_10j
IdentityIdentityRelu_10:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
��
�
F__inference_model_6_layer_call_and_return_conditional_losses_203567098

inputs<
8batch_normalization_15_batchnorm_readvariableop_resource@
<batch_normalization_15_batchnorm_mul_readvariableop_resource>
:batch_normalization_15_batchnorm_readvariableop_1_resource>
:batch_normalization_15_batchnorm_readvariableop_2_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource<
8batch_normalization_16_batchnorm_readvariableop_resource@
<batch_normalization_16_batchnorm_mul_readvariableop_resource>
:batch_normalization_16_batchnorm_readvariableop_1_resource>
:batch_normalization_16_batchnorm_readvariableop_2_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource<
8batch_normalization_17_batchnorm_readvariableop_resource@
<batch_normalization_17_batchnorm_mul_readvariableop_resource>
:batch_normalization_17_batchnorm_readvariableop_1_resource>
:batch_normalization_17_batchnorm_readvariableop_2_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity��/batch_normalization_15/batchnorm/ReadVariableOp�1batch_normalization_15/batchnorm/ReadVariableOp_1�1batch_normalization_15/batchnorm/ReadVariableOp_2�3batch_normalization_15/batchnorm/mul/ReadVariableOp�/batch_normalization_16/batchnorm/ReadVariableOp�1batch_normalization_16/batchnorm/ReadVariableOp_1�1batch_normalization_16/batchnorm/ReadVariableOp_2�3batch_normalization_16/batchnorm/mul/ReadVariableOp�/batch_normalization_17/batchnorm/ReadVariableOp�1batch_normalization_17/batchnorm/ReadVariableOp_1�1batch_normalization_17/batchnorm/ReadVariableOp_2�3batch_normalization_17/batchnorm/mul/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�
#batch_normalization_15/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_15/LogicalAnd/x�
#batch_normalization_15/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_15/LogicalAnd/y�
!batch_normalization_15/LogicalAnd
LogicalAnd,batch_normalization_15/LogicalAnd/x:output:0,batch_normalization_15/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_15/LogicalAnd�
/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_15/batchnorm/ReadVariableOp�
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_15/batchnorm/add/y�
$batch_normalization_15/batchnorm/addAddV27batch_normalization_15/batchnorm/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt�
3batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_15/batchnorm/mul/ReadVariableOp�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:0;batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul�
&batch_normalization_15/batchnorm/mul_1Mulinputs(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/mul_1�
1batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_1�
&batch_normalization_15/batchnorm/mul_2Mul9batch_normalization_15/batchnorm/ReadVariableOp_1:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2�
1batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_15/batchnorm/ReadVariableOp_2�
$batch_normalization_15/batchnorm/subSub9batch_normalization_15/batchnorm/ReadVariableOp_2:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/add_1�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMul*batch_normalization_15/batchnorm/add_1:z:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_18/BiasAdd�
#batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_16/LogicalAnd/x�
#batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/y�
!batch_normalization_16/LogicalAnd
LogicalAnd,batch_normalization_16/LogicalAnd/x:output:0,batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_16/LogicalAnd�
/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype021
/batch_normalization_16/batchnorm/ReadVariableOp�
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_16/batchnorm/add/y�
$batch_normalization_16/batchnorm/addAddV27batch_normalization_16/batchnorm/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_16/batchnorm/add�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_16/batchnorm/Rsqrt�
3batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype025
3batch_normalization_16/batchnorm/mul/ReadVariableOp�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:0;batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_16/batchnorm/mul�
&batch_normalization_16/batchnorm/mul_1Muldense_18/BiasAdd:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_16/batchnorm/mul_1�
1batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_1�
&batch_normalization_16/batchnorm/mul_2Mul9batch_normalization_16/batchnorm/ReadVariableOp_1:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_16/batchnorm/mul_2�
1batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype023
1batch_normalization_16/batchnorm/ReadVariableOp_2�
$batch_normalization_16/batchnorm/subSub9batch_normalization_16/batchnorm/ReadVariableOp_2:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_16/batchnorm/sub�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_16/batchnorm/add_1�
tf_op_layer_Relu_9/Relu_9Relu*batch_normalization_16/batchnorm/add_1:z:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Relu_9/Relu_9�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMul'tf_op_layer_Relu_9/Relu_9:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_19/BiasAdd�
#batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_17/LogicalAnd/x�
#batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/y�
!batch_normalization_17/LogicalAnd
LogicalAnd,batch_normalization_17/LogicalAnd/x:output:0,batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_17/LogicalAnd�
/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype021
/batch_normalization_17/batchnorm/ReadVariableOp�
&batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2(
&batch_normalization_17/batchnorm/add/y�
$batch_normalization_17/batchnorm/addAddV27batch_normalization_17/batchnorm/ReadVariableOp:value:0/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_17/batchnorm/add�
&batch_normalization_17/batchnorm/RsqrtRsqrt(batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_17/batchnorm/Rsqrt�
3batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype025
3batch_normalization_17/batchnorm/mul/ReadVariableOp�
$batch_normalization_17/batchnorm/mulMul*batch_normalization_17/batchnorm/Rsqrt:y:0;batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_17/batchnorm/mul�
&batch_normalization_17/batchnorm/mul_1Muldense_19/BiasAdd:output:0(batch_normalization_17/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_17/batchnorm/mul_1�
1batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_1�
&batch_normalization_17/batchnorm/mul_2Mul9batch_normalization_17/batchnorm/ReadVariableOp_1:value:0(batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_17/batchnorm/mul_2�
1batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype023
1batch_normalization_17/batchnorm/ReadVariableOp_2�
$batch_normalization_17/batchnorm/subSub9batch_normalization_17/batchnorm/ReadVariableOp_2:value:0*batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_17/batchnorm/sub�
&batch_normalization_17/batchnorm/add_1AddV2*batch_normalization_17/batchnorm/mul_1:z:0(batch_normalization_17/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_17/batchnorm/add_1�
tf_op_layer_Relu_10/Relu_10Relu*batch_normalization_17/batchnorm/add_1:z:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Relu_10/Relu_10�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul)tf_op_layer_Relu_10/Relu_10:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdds
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_20/Tanh
tf_op_layer_Mul_3/Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB 2      �?2
tf_op_layer_Mul_3/Mul_3/y�
tf_op_layer_Mul_3/Mul_3Muldense_20/Tanh:y:0"tf_op_layer_Mul_3/Mul_3/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������2
tf_op_layer_Mul_3/Mul_3�
IdentityIdentitytf_op_layer_Mul_3/Mul_3:z:00^batch_normalization_15/batchnorm/ReadVariableOp2^batch_normalization_15/batchnorm/ReadVariableOp_12^batch_normalization_15/batchnorm/ReadVariableOp_24^batch_normalization_15/batchnorm/mul/ReadVariableOp0^batch_normalization_16/batchnorm/ReadVariableOp2^batch_normalization_16/batchnorm/ReadVariableOp_12^batch_normalization_16/batchnorm/ReadVariableOp_24^batch_normalization_16/batchnorm/mul/ReadVariableOp0^batch_normalization_17/batchnorm/ReadVariableOp2^batch_normalization_17/batchnorm/ReadVariableOp_12^batch_normalization_17/batchnorm/ReadVariableOp_24^batch_normalization_17/batchnorm/mul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::2b
/batch_normalization_15/batchnorm/ReadVariableOp/batch_normalization_15/batchnorm/ReadVariableOp2f
1batch_normalization_15/batchnorm/ReadVariableOp_11batch_normalization_15/batchnorm/ReadVariableOp_12f
1batch_normalization_15/batchnorm/ReadVariableOp_21batch_normalization_15/batchnorm/ReadVariableOp_22j
3batch_normalization_15/batchnorm/mul/ReadVariableOp3batch_normalization_15/batchnorm/mul/ReadVariableOp2b
/batch_normalization_16/batchnorm/ReadVariableOp/batch_normalization_16/batchnorm/ReadVariableOp2f
1batch_normalization_16/batchnorm/ReadVariableOp_11batch_normalization_16/batchnorm/ReadVariableOp_12f
1batch_normalization_16/batchnorm/ReadVariableOp_21batch_normalization_16/batchnorm/ReadVariableOp_22j
3batch_normalization_16/batchnorm/mul/ReadVariableOp3batch_normalization_16/batchnorm/mul/ReadVariableOp2b
/batch_normalization_17/batchnorm/ReadVariableOp/batch_normalization_17/batchnorm/ReadVariableOp2f
1batch_normalization_17/batchnorm/ReadVariableOp_11batch_normalization_17/batchnorm/ReadVariableOp_12f
1batch_normalization_17/batchnorm/ReadVariableOp_21batch_normalization_17/batchnorm/ReadVariableOp_22j
3batch_normalization_17/batchnorm/mul/ReadVariableOp3batch_normalization_17/batchnorm/mul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203567246

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
G__inference_dense_18_layer_call_and_return_conditional_losses_203567274

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203567530

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
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
+__inference_model_6_layer_call_fn_203566684
input_10"
statefulpartitionedcall_args_1"
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
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_10statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_2035666632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
input_10
�
�
:__inference_batch_normalization_16_layer_call_fn_203567401

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
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2035662622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
S
5__inference_tf_op_layer_Mul_3_layer_call_fn_203567587
inputs_0
identity�
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_2035665892
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:( $
"
_user_specified_name
inputs/0
�
U
7__inference_tf_op_layer_Relu_10_layer_call_fn_203567558
inputs_0
identity�
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_2035665522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:( $
"
_user_specified_name
inputs/0
�
n
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_203567582
inputs_0
identity[
Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB 2      �?2	
Mul_3/yr
Mul_3Mulinputs_0Mul_3/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������2
Mul_3]
IdentityIdentity	Mul_3:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:( $
"
_user_specified_name
inputs/0
�
�
+__inference_model_6_layer_call_fn_203566738
input_10"
statefulpartitionedcall_args_1"
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
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_10statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_2035667172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
input_10
�P
�

%__inference__traced_restore_203567731
file_prefix1
-assignvariableop_batch_normalization_15_gamma2
.assignvariableop_1_batch_normalization_15_beta9
5assignvariableop_2_batch_normalization_15_moving_mean=
9assignvariableop_3_batch_normalization_15_moving_variance&
"assignvariableop_4_dense_18_kernel$
 assignvariableop_5_dense_18_bias3
/assignvariableop_6_batch_normalization_16_gamma2
.assignvariableop_7_batch_normalization_16_beta9
5assignvariableop_8_batch_normalization_16_moving_mean=
9assignvariableop_9_batch_normalization_16_moving_variance'
#assignvariableop_10_dense_19_kernel%
!assignvariableop_11_dense_19_bias4
0assignvariableop_12_batch_normalization_17_gamma3
/assignvariableop_13_batch_normalization_17_beta:
6assignvariableop_14_batch_normalization_17_moving_mean>
:assignvariableop_15_batch_normalization_17_moving_variance'
#assignvariableop_16_dense_20_kernel%
!assignvariableop_17_dense_20_bias
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp-assignvariableop_batch_normalization_15_gammaIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_15_betaIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp5assignvariableop_2_batch_normalization_15_moving_meanIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_normalization_15_moving_varianceIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_18_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_18_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_16_gammaIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_16_betaIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_16_moving_meanIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_16_moving_varianceIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_19_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_19_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_17_gammaIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_17_betaIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_17_moving_meanIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_17_moving_varianceIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_20_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_20_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17�
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
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18�
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
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
�2
�
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203567360

inputs
assignmovingavg_203567333
assignmovingavg_1_203567340)
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/203567333*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/203567333*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203567333*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/203567333*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/203567333*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203567333AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/203567333*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/203567340*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/203567340*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203567340*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203567340*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203567340*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203567340AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/203567340*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�2
�
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203567223

inputs
assignmovingavg_203567196
assignmovingavg_1_203567203)
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
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/203567196*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/203567196*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203567196*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/203567196*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/203567196*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203567196AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/203567196*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/203567203*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/203567203*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203567203*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203567203*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203567203*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203567203AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/203567203*
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
�
�
:__inference_batch_normalization_16_layer_call_fn_203567392

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
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_2035662302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
:__inference_batch_normalization_17_layer_call_fn_203567548

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
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2035664102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
:__inference_batch_normalization_15_layer_call_fn_203567264

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
GPU2*0J 8*^
fYRW
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_2035661142
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
�
�
+__inference_model_6_layer_call_fn_203567144

inputs"
statefulpartitionedcall_args_1"
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
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_2035667172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
,__inference_dense_18_layer_call_fn_203567281

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2035664542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
��
�
$__inference__wrapped_model_203565973
input_10D
@model_6_batch_normalization_15_batchnorm_readvariableop_resourceH
Dmodel_6_batch_normalization_15_batchnorm_mul_readvariableop_resourceF
Bmodel_6_batch_normalization_15_batchnorm_readvariableop_1_resourceF
Bmodel_6_batch_normalization_15_batchnorm_readvariableop_2_resource3
/model_6_dense_18_matmul_readvariableop_resource4
0model_6_dense_18_biasadd_readvariableop_resourceD
@model_6_batch_normalization_16_batchnorm_readvariableop_resourceH
Dmodel_6_batch_normalization_16_batchnorm_mul_readvariableop_resourceF
Bmodel_6_batch_normalization_16_batchnorm_readvariableop_1_resourceF
Bmodel_6_batch_normalization_16_batchnorm_readvariableop_2_resource3
/model_6_dense_19_matmul_readvariableop_resource4
0model_6_dense_19_biasadd_readvariableop_resourceD
@model_6_batch_normalization_17_batchnorm_readvariableop_resourceH
Dmodel_6_batch_normalization_17_batchnorm_mul_readvariableop_resourceF
Bmodel_6_batch_normalization_17_batchnorm_readvariableop_1_resourceF
Bmodel_6_batch_normalization_17_batchnorm_readvariableop_2_resource3
/model_6_dense_20_matmul_readvariableop_resource4
0model_6_dense_20_biasadd_readvariableop_resource
identity��7model_6/batch_normalization_15/batchnorm/ReadVariableOp�9model_6/batch_normalization_15/batchnorm/ReadVariableOp_1�9model_6/batch_normalization_15/batchnorm/ReadVariableOp_2�;model_6/batch_normalization_15/batchnorm/mul/ReadVariableOp�7model_6/batch_normalization_16/batchnorm/ReadVariableOp�9model_6/batch_normalization_16/batchnorm/ReadVariableOp_1�9model_6/batch_normalization_16/batchnorm/ReadVariableOp_2�;model_6/batch_normalization_16/batchnorm/mul/ReadVariableOp�7model_6/batch_normalization_17/batchnorm/ReadVariableOp�9model_6/batch_normalization_17/batchnorm/ReadVariableOp_1�9model_6/batch_normalization_17/batchnorm/ReadVariableOp_2�;model_6/batch_normalization_17/batchnorm/mul/ReadVariableOp�'model_6/dense_18/BiasAdd/ReadVariableOp�&model_6/dense_18/MatMul/ReadVariableOp�'model_6/dense_19/BiasAdd/ReadVariableOp�&model_6/dense_19/MatMul/ReadVariableOp�'model_6/dense_20/BiasAdd/ReadVariableOp�&model_6/dense_20/MatMul/ReadVariableOp�
+model_6/batch_normalization_15/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+model_6/batch_normalization_15/LogicalAnd/x�
+model_6/batch_normalization_15/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2-
+model_6/batch_normalization_15/LogicalAnd/y�
)model_6/batch_normalization_15/LogicalAnd
LogicalAnd4model_6/batch_normalization_15/LogicalAnd/x:output:04model_6/batch_normalization_15/LogicalAnd/y:output:0*
_output_shapes
: 2+
)model_6/batch_normalization_15/LogicalAnd�
7model_6/batch_normalization_15/batchnorm/ReadVariableOpReadVariableOp@model_6_batch_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype029
7model_6/batch_normalization_15/batchnorm/ReadVariableOp�
.model_6/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?20
.model_6/batch_normalization_15/batchnorm/add/y�
,model_6/batch_normalization_15/batchnorm/addAddV2?model_6/batch_normalization_15/batchnorm/ReadVariableOp:value:07model_6/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2.
,model_6/batch_normalization_15/batchnorm/add�
.model_6/batch_normalization_15/batchnorm/RsqrtRsqrt0model_6/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:20
.model_6/batch_normalization_15/batchnorm/Rsqrt�
;model_6/batch_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_batch_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02=
;model_6/batch_normalization_15/batchnorm/mul/ReadVariableOp�
,model_6/batch_normalization_15/batchnorm/mulMul2model_6/batch_normalization_15/batchnorm/Rsqrt:y:0Cmodel_6/batch_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,model_6/batch_normalization_15/batchnorm/mul�
.model_6/batch_normalization_15/batchnorm/mul_1Mulinput_100model_6/batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������20
.model_6/batch_normalization_15/batchnorm/mul_1�
9model_6/batch_normalization_15/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_6_batch_normalization_15_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9model_6/batch_normalization_15/batchnorm/ReadVariableOp_1�
.model_6/batch_normalization_15/batchnorm/mul_2MulAmodel_6/batch_normalization_15/batchnorm/ReadVariableOp_1:value:00model_6/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:20
.model_6/batch_normalization_15/batchnorm/mul_2�
9model_6/batch_normalization_15/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_6_batch_normalization_15_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02;
9model_6/batch_normalization_15/batchnorm/ReadVariableOp_2�
,model_6/batch_normalization_15/batchnorm/subSubAmodel_6/batch_normalization_15/batchnorm/ReadVariableOp_2:value:02model_6/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2.
,model_6/batch_normalization_15/batchnorm/sub�
.model_6/batch_normalization_15/batchnorm/add_1AddV22model_6/batch_normalization_15/batchnorm/mul_1:z:00model_6/batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������20
.model_6/batch_normalization_15/batchnorm/add_1�
&model_6/dense_18/MatMul/ReadVariableOpReadVariableOp/model_6_dense_18_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02(
&model_6/dense_18/MatMul/ReadVariableOp�
model_6/dense_18/MatMulMatMul2model_6/batch_normalization_15/batchnorm/add_1:z:0.model_6/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_6/dense_18/MatMul�
'model_6/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_6/dense_18/BiasAdd/ReadVariableOp�
model_6/dense_18/BiasAddBiasAdd!model_6/dense_18/MatMul:product:0/model_6/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_6/dense_18/BiasAdd�
+model_6/batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+model_6/batch_normalization_16/LogicalAnd/x�
+model_6/batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2-
+model_6/batch_normalization_16/LogicalAnd/y�
)model_6/batch_normalization_16/LogicalAnd
LogicalAnd4model_6/batch_normalization_16/LogicalAnd/x:output:04model_6/batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2+
)model_6/batch_normalization_16/LogicalAnd�
7model_6/batch_normalization_16/batchnorm/ReadVariableOpReadVariableOp@model_6_batch_normalization_16_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype029
7model_6/batch_normalization_16/batchnorm/ReadVariableOp�
.model_6/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?20
.model_6/batch_normalization_16/batchnorm/add/y�
,model_6/batch_normalization_16/batchnorm/addAddV2?model_6/batch_normalization_16/batchnorm/ReadVariableOp:value:07model_6/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2.
,model_6/batch_normalization_16/batchnorm/add�
.model_6/batch_normalization_16/batchnorm/RsqrtRsqrt0model_6/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes	
:�20
.model_6/batch_normalization_16/batchnorm/Rsqrt�
;model_6/batch_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_batch_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;model_6/batch_normalization_16/batchnorm/mul/ReadVariableOp�
,model_6/batch_normalization_16/batchnorm/mulMul2model_6/batch_normalization_16/batchnorm/Rsqrt:y:0Cmodel_6/batch_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,model_6/batch_normalization_16/batchnorm/mul�
.model_6/batch_normalization_16/batchnorm/mul_1Mul!model_6/dense_18/BiasAdd:output:00model_6/batch_normalization_16/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������20
.model_6/batch_normalization_16/batchnorm/mul_1�
9model_6/batch_normalization_16/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_6_batch_normalization_16_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9model_6/batch_normalization_16/batchnorm/ReadVariableOp_1�
.model_6/batch_normalization_16/batchnorm/mul_2MulAmodel_6/batch_normalization_16/batchnorm/ReadVariableOp_1:value:00model_6/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes	
:�20
.model_6/batch_normalization_16/batchnorm/mul_2�
9model_6/batch_normalization_16/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_6_batch_normalization_16_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02;
9model_6/batch_normalization_16/batchnorm/ReadVariableOp_2�
,model_6/batch_normalization_16/batchnorm/subSubAmodel_6/batch_normalization_16/batchnorm/ReadVariableOp_2:value:02model_6/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2.
,model_6/batch_normalization_16/batchnorm/sub�
.model_6/batch_normalization_16/batchnorm/add_1AddV22model_6/batch_normalization_16/batchnorm/mul_1:z:00model_6/batch_normalization_16/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������20
.model_6/batch_normalization_16/batchnorm/add_1�
!model_6/tf_op_layer_Relu_9/Relu_9Relu2model_6/batch_normalization_16/batchnorm/add_1:z:0*
T0*
_cloned(*(
_output_shapes
:����������2#
!model_6/tf_op_layer_Relu_9/Relu_9�
&model_6/dense_19/MatMul/ReadVariableOpReadVariableOp/model_6_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&model_6/dense_19/MatMul/ReadVariableOp�
model_6/dense_19/MatMulMatMul/model_6/tf_op_layer_Relu_9/Relu_9:activations:0.model_6/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_6/dense_19/MatMul�
'model_6/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_6/dense_19/BiasAdd/ReadVariableOp�
model_6/dense_19/BiasAddBiasAdd!model_6/dense_19/MatMul:product:0/model_6/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_6/dense_19/BiasAdd�
+model_6/batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+model_6/batch_normalization_17/LogicalAnd/x�
+model_6/batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2-
+model_6/batch_normalization_17/LogicalAnd/y�
)model_6/batch_normalization_17/LogicalAnd
LogicalAnd4model_6/batch_normalization_17/LogicalAnd/x:output:04model_6/batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2+
)model_6/batch_normalization_17/LogicalAnd�
7model_6/batch_normalization_17/batchnorm/ReadVariableOpReadVariableOp@model_6_batch_normalization_17_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype029
7model_6/batch_normalization_17/batchnorm/ReadVariableOp�
.model_6/batch_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?20
.model_6/batch_normalization_17/batchnorm/add/y�
,model_6/batch_normalization_17/batchnorm/addAddV2?model_6/batch_normalization_17/batchnorm/ReadVariableOp:value:07model_6/batch_normalization_17/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2.
,model_6/batch_normalization_17/batchnorm/add�
.model_6/batch_normalization_17/batchnorm/RsqrtRsqrt0model_6/batch_normalization_17/batchnorm/add:z:0*
T0*
_output_shapes	
:�20
.model_6/batch_normalization_17/batchnorm/Rsqrt�
;model_6/batch_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_6_batch_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;model_6/batch_normalization_17/batchnorm/mul/ReadVariableOp�
,model_6/batch_normalization_17/batchnorm/mulMul2model_6/batch_normalization_17/batchnorm/Rsqrt:y:0Cmodel_6/batch_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,model_6/batch_normalization_17/batchnorm/mul�
.model_6/batch_normalization_17/batchnorm/mul_1Mul!model_6/dense_19/BiasAdd:output:00model_6/batch_normalization_17/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������20
.model_6/batch_normalization_17/batchnorm/mul_1�
9model_6/batch_normalization_17/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_6_batch_normalization_17_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9model_6/batch_normalization_17/batchnorm/ReadVariableOp_1�
.model_6/batch_normalization_17/batchnorm/mul_2MulAmodel_6/batch_normalization_17/batchnorm/ReadVariableOp_1:value:00model_6/batch_normalization_17/batchnorm/mul:z:0*
T0*
_output_shapes	
:�20
.model_6/batch_normalization_17/batchnorm/mul_2�
9model_6/batch_normalization_17/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_6_batch_normalization_17_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02;
9model_6/batch_normalization_17/batchnorm/ReadVariableOp_2�
,model_6/batch_normalization_17/batchnorm/subSubAmodel_6/batch_normalization_17/batchnorm/ReadVariableOp_2:value:02model_6/batch_normalization_17/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2.
,model_6/batch_normalization_17/batchnorm/sub�
.model_6/batch_normalization_17/batchnorm/add_1AddV22model_6/batch_normalization_17/batchnorm/mul_1:z:00model_6/batch_normalization_17/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������20
.model_6/batch_normalization_17/batchnorm/add_1�
#model_6/tf_op_layer_Relu_10/Relu_10Relu2model_6/batch_normalization_17/batchnorm/add_1:z:0*
T0*
_cloned(*(
_output_shapes
:����������2%
#model_6/tf_op_layer_Relu_10/Relu_10�
&model_6/dense_20/MatMul/ReadVariableOpReadVariableOp/model_6_dense_20_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02(
&model_6/dense_20/MatMul/ReadVariableOp�
model_6/dense_20/MatMulMatMul1model_6/tf_op_layer_Relu_10/Relu_10:activations:0.model_6/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_6/dense_20/MatMul�
'model_6/dense_20/BiasAdd/ReadVariableOpReadVariableOp0model_6_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_6/dense_20/BiasAdd/ReadVariableOp�
model_6/dense_20/BiasAddBiasAdd!model_6/dense_20/MatMul:product:0/model_6/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_6/dense_20/BiasAdd�
model_6/dense_20/TanhTanh!model_6/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model_6/dense_20/Tanh�
!model_6/tf_op_layer_Mul_3/Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB 2      �?2#
!model_6/tf_op_layer_Mul_3/Mul_3/y�
model_6/tf_op_layer_Mul_3/Mul_3Mulmodel_6/dense_20/Tanh:y:0*model_6/tf_op_layer_Mul_3/Mul_3/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������2!
model_6/tf_op_layer_Mul_3/Mul_3�
IdentityIdentity#model_6/tf_op_layer_Mul_3/Mul_3:z:08^model_6/batch_normalization_15/batchnorm/ReadVariableOp:^model_6/batch_normalization_15/batchnorm/ReadVariableOp_1:^model_6/batch_normalization_15/batchnorm/ReadVariableOp_2<^model_6/batch_normalization_15/batchnorm/mul/ReadVariableOp8^model_6/batch_normalization_16/batchnorm/ReadVariableOp:^model_6/batch_normalization_16/batchnorm/ReadVariableOp_1:^model_6/batch_normalization_16/batchnorm/ReadVariableOp_2<^model_6/batch_normalization_16/batchnorm/mul/ReadVariableOp8^model_6/batch_normalization_17/batchnorm/ReadVariableOp:^model_6/batch_normalization_17/batchnorm/ReadVariableOp_1:^model_6/batch_normalization_17/batchnorm/ReadVariableOp_2<^model_6/batch_normalization_17/batchnorm/mul/ReadVariableOp(^model_6/dense_18/BiasAdd/ReadVariableOp'^model_6/dense_18/MatMul/ReadVariableOp(^model_6/dense_19/BiasAdd/ReadVariableOp'^model_6/dense_19/MatMul/ReadVariableOp(^model_6/dense_20/BiasAdd/ReadVariableOp'^model_6/dense_20/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::2r
7model_6/batch_normalization_15/batchnorm/ReadVariableOp7model_6/batch_normalization_15/batchnorm/ReadVariableOp2v
9model_6/batch_normalization_15/batchnorm/ReadVariableOp_19model_6/batch_normalization_15/batchnorm/ReadVariableOp_12v
9model_6/batch_normalization_15/batchnorm/ReadVariableOp_29model_6/batch_normalization_15/batchnorm/ReadVariableOp_22z
;model_6/batch_normalization_15/batchnorm/mul/ReadVariableOp;model_6/batch_normalization_15/batchnorm/mul/ReadVariableOp2r
7model_6/batch_normalization_16/batchnorm/ReadVariableOp7model_6/batch_normalization_16/batchnorm/ReadVariableOp2v
9model_6/batch_normalization_16/batchnorm/ReadVariableOp_19model_6/batch_normalization_16/batchnorm/ReadVariableOp_12v
9model_6/batch_normalization_16/batchnorm/ReadVariableOp_29model_6/batch_normalization_16/batchnorm/ReadVariableOp_22z
;model_6/batch_normalization_16/batchnorm/mul/ReadVariableOp;model_6/batch_normalization_16/batchnorm/mul/ReadVariableOp2r
7model_6/batch_normalization_17/batchnorm/ReadVariableOp7model_6/batch_normalization_17/batchnorm/ReadVariableOp2v
9model_6/batch_normalization_17/batchnorm/ReadVariableOp_19model_6/batch_normalization_17/batchnorm/ReadVariableOp_12v
9model_6/batch_normalization_17/batchnorm/ReadVariableOp_29model_6/batch_normalization_17/batchnorm/ReadVariableOp_22z
;model_6/batch_normalization_17/batchnorm/mul/ReadVariableOp;model_6/batch_normalization_17/batchnorm/mul/ReadVariableOp2R
'model_6/dense_18/BiasAdd/ReadVariableOp'model_6/dense_18/BiasAdd/ReadVariableOp2P
&model_6/dense_18/MatMul/ReadVariableOp&model_6/dense_18/MatMul/ReadVariableOp2R
'model_6/dense_19/BiasAdd/ReadVariableOp'model_6/dense_19/BiasAdd/ReadVariableOp2P
&model_6/dense_19/MatMul/ReadVariableOp&model_6/dense_19/MatMul/ReadVariableOp2R
'model_6/dense_20/BiasAdd/ReadVariableOp'model_6/dense_20/BiasAdd/ReadVariableOp2P
&model_6/dense_20/MatMul/ReadVariableOp&model_6/dense_20/MatMul/ReadVariableOp:( $
"
_user_specified_name
input_10
�
�
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203566114

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
�2
�
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203566378

inputs
assignmovingavg_203566351
assignmovingavg_1_203566358)
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/203566351*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/203566351*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203566351*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/203566351*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/203566351*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203566351AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/203566351*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/203566358*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/203566358*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203566358*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203566358*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203566358*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203566358AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/203566358*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
l
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_203566589

inputs
identity[
Mul_3/yConst*
_output_shapes
: *
dtype0*
valueB 2      �?2	
Mul_3/yp
Mul_3MulinputsMul_3/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������2
Mul_3]
IdentityIdentity	Mul_3:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
'__inference_signature_wrapper_203566876
input_10"
statefulpartitionedcall_args_1"
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
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_10statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
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
$__inference__wrapped_model_2035659732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
input_10
�	
�
G__inference_dense_20_layer_call_and_return_conditional_losses_203566571

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
:__inference_batch_normalization_17_layer_call_fn_203567539

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
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_2035663782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
+__inference_model_6_layer_call_fn_203567121

inputs"
statefulpartitionedcall_args_1"
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
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_2035666632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
G__inference_dense_18_layer_call_and_return_conditional_losses_203566454

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
m
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_203566494

inputs
identityb
Relu_9Reluinputs*
T0*
_cloned(*(
_output_shapes
:����������2
Relu_9i
IdentityIdentityRelu_9:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
,__inference_dense_19_layer_call_fn_203567428

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2035665122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�6
�

F__inference_model_6_layer_call_and_return_conditional_losses_203566598
input_109
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_4+
'dense_18_statefulpartitionedcall_args_1+
'dense_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_4+
'dense_19_statefulpartitionedcall_args_1+
'dense_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_4+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallinput_105batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
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
GPU2*0J 8*^
fYRW
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_20356608220
.batch_normalization_15/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0'dense_18_statefulpartitionedcall_args_1'dense_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2035664542"
 dense_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_20356623020
.batch_normalization_16/StatefulPartitionedCall�
"tf_op_layer_Relu_9/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_2035664942$
"tf_op_layer_Relu_9/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_Relu_9/PartitionedCall:output:0'dense_19_statefulpartitionedcall_args_1'dense_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2035665122"
 dense_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_20356637820
.batch_normalization_17/StatefulPartitionedCall�
#tf_op_layer_Relu_10/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_2035665522%
#tf_op_layer_Relu_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Relu_10/PartitionedCall:output:0'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2035665712"
 dense_20/StatefulPartitionedCall�
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_2035665892#
!tf_op_layer_Mul_3/PartitionedCall�
IdentityIdentity*tf_op_layer_Mul_3/PartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:( $
"
_user_specified_name
input_10
�
p
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_203567553
inputs_0
identityf
Relu_10Reluinputs_0*
T0*
_cloned(*(
_output_shapes
:����������2	
Relu_10j
IdentityIdentityRelu_10:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:( $
"
_user_specified_name
inputs/0
�6
�

F__inference_model_6_layer_call_and_return_conditional_losses_203566717

inputs9
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_4+
'dense_18_statefulpartitionedcall_args_1+
'dense_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_4+
'dense_19_statefulpartitionedcall_args_1+
'dense_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_4+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallinputs5batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
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
GPU2*0J 8*^
fYRW
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_20356611420
.batch_normalization_15/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0'dense_18_statefulpartitionedcall_args_1'dense_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2035664542"
 dense_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_20356626220
.batch_normalization_16/StatefulPartitionedCall�
"tf_op_layer_Relu_9/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_2035664942$
"tf_op_layer_Relu_9/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_Relu_9/PartitionedCall:output:0'dense_19_statefulpartitionedcall_args_1'dense_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2035665122"
 dense_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_20356641020
.batch_normalization_17/StatefulPartitionedCall�
#tf_op_layer_Relu_10/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_2035665522%
#tf_op_layer_Relu_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Relu_10/PartitionedCall:output:0'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2035665712"
 dense_20/StatefulPartitionedCall�
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_2035665892#
!tf_op_layer_Mul_3/PartitionedCall�
IdentityIdentity*tf_op_layer_Mul_3/PartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�2
�
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203566230

inputs
assignmovingavg_203566203
assignmovingavg_1_203566210)
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/203566203*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/203566203*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203566203*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/203566203*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/203566203*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203566203AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/203566203*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/203566210*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/203566210*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203566210*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203566210*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203566210*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203566210AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/203566210*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
G__inference_dense_20_layer_call_and_return_conditional_losses_203567569

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
o
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_203567406
inputs_0
identityd
Relu_9Reluinputs_0*
T0*
_cloned(*(
_output_shapes
:����������2
Relu_9i
IdentityIdentityRelu_9:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:( $
"
_user_specified_name
inputs/0
�
�
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203566410

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
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�2
�
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203566082

inputs
assignmovingavg_203566055
assignmovingavg_1_203566062)
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
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/203566055*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/203566055*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203566055*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/203566055*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/203566055*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203566055AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/203566055*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/203566062*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/203566062*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203566062*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203566062*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203566062*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203566062AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/203566062*
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
,__inference_dense_20_layer_call_fn_203567576

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
GPU2*0J 8*P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2035665712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�2
�
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203567507

inputs
assignmovingavg_203567480
assignmovingavg_1_203567487)
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
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
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
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*,
_class"
 loc:@AssignMovingAvg/203567480*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*,
_class"
 loc:@AssignMovingAvg/203567480*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_203567480*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*,
_class"
 loc:@AssignMovingAvg/203567480*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*,
_class"
 loc:@AssignMovingAvg/203567480*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_203567480AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg/203567480*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*.
_class$
" loc:@AssignMovingAvg_1/203567487*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*.
_class$
" loc:@AssignMovingAvg_1/203567487*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_203567487*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203567487*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/203567487*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_203567487AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*.
_class$
" loc:@AssignMovingAvg_1/203567487*
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
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�6
�

F__inference_model_6_layer_call_and_return_conditional_losses_203566663

inputs9
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_4+
'dense_18_statefulpartitionedcall_args_1+
'dense_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_4+
'dense_19_statefulpartitionedcall_args_1+
'dense_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_4+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallinputs5batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
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
GPU2*0J 8*^
fYRW
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_20356608220
.batch_normalization_15/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0'dense_18_statefulpartitionedcall_args_1'dense_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2035664542"
 dense_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_20356623020
.batch_normalization_16/StatefulPartitionedCall�
"tf_op_layer_Relu_9/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_2035664942$
"tf_op_layer_Relu_9/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_Relu_9/PartitionedCall:output:0'dense_19_statefulpartitionedcall_args_1'dense_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2035665122"
 dense_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_20356637820
.batch_normalization_17/StatefulPartitionedCall�
#tf_op_layer_Relu_10/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_2035665522%
#tf_op_layer_Relu_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Relu_10/PartitionedCall:output:0'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2035665712"
 dense_20/StatefulPartitionedCall�
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_2035665892#
!tf_op_layer_Mul_3/PartitionedCall�
IdentityIdentity*tf_op_layer_Mul_3/PartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203567383

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
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2����MbP?2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs
�6
�

F__inference_model_6_layer_call_and_return_conditional_losses_203566629
input_109
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_4+
'dense_18_statefulpartitionedcall_args_1+
'dense_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_4+
'dense_19_statefulpartitionedcall_args_1+
'dense_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_4+
'dense_20_statefulpartitionedcall_args_1+
'dense_20_statefulpartitionedcall_args_2
identity��.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCallinput_105batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
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
GPU2*0J 8*^
fYRW
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_20356611420
.batch_normalization_15/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0'dense_18_statefulpartitionedcall_args_1'dense_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2035664542"
 dense_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_20356626220
.batch_normalization_16/StatefulPartitionedCall�
"tf_op_layer_Relu_9/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_2035664942$
"tf_op_layer_Relu_9/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+tf_op_layer_Relu_9/PartitionedCall:output:0'dense_19_statefulpartitionedcall_args_1'dense_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2035665122"
 dense_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*^
fYRW
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_20356641020
.batch_normalization_17/StatefulPartitionedCall�
#tf_op_layer_Relu_10/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*[
fVRT
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_2035665522%
#tf_op_layer_Relu_10/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Relu_10/PartitionedCall:output:0'dense_20_statefulpartitionedcall_args_1'dense_20_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2035665712"
 dense_20/StatefulPartitionedCall�
!tf_op_layer_Mul_3/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_2035665892#
!tf_op_layer_Mul_3/PartitionedCall�
IdentityIdentity*tf_op_layer_Mul_3/PartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:���������::::::::::::::::::2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:( $
"
_user_specified_name
input_10"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_101
serving_default_input_10:0���������E
tf_op_layer_Mul_30
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�W
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
u__call__
*v&call_and_return_all_conditional_losses
w_default_save_signature"�S
_tf_keras_model�S{"class_name": "Model", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float64", "units": 400, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_9", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_9", "op": "Relu", "input": ["batch_normalization_16/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}, "name": "tf_op_layer_Relu_9", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float64", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["tf_op_layer_Relu_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_10", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_10", "op": "Relu", "input": ["batch_normalization_17/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}, "name": "tf_op_layer_Relu_10", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float64", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.003, "maxval": 0.003, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.003, "maxval": 0.003, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["tf_op_layer_Relu_10", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_3", "trainable": true, "dtype": "float64", "node_def": {"name": "Mul_3", "op": "Mul", "input": ["dense_20/Identity", "Mul_3/y"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_Mul_3", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["tf_op_layer_Mul_3", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_15", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float64", "units": 400, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["batch_normalization_15", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_16", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_9", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_9", "op": "Relu", "input": ["batch_normalization_16/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}, "name": "tf_op_layer_Relu_9", "inbound_nodes": [[["batch_normalization_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float64", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["tf_op_layer_Relu_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_17", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_10", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_10", "op": "Relu", "input": ["batch_normalization_17/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}, "name": "tf_op_layer_Relu_10", "inbound_nodes": [[["batch_normalization_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float64", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.003, "maxval": 0.003, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.003, "maxval": 0.003, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["tf_op_layer_Relu_10", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_3", "trainable": true, "dtype": "float64", "node_def": {"name": "Mul_3", "op": "Mul", "input": ["dense_20/Identity", "Mul_3/y"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_Mul_3", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["tf_op_layer_Mul_3", 0, 0]]}}, "training_config": {"loss": "MSE", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_10", "dtype": "float64", "sparse": false, "ragged": false, "batch_input_shape": [null, 2], "config": {"batch_input_shape": [null, 2], "dtype": "float64", "sparse": false, "ragged": false, "name": "input_10"}}
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 2}}}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "dense_18", "trainable": true, "dtype": "float64", "units": 400, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.7071067811865475, "maxval": 0.7071067811865475, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
�
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%	variables
&trainable_variables
'regularization_losses
(	keras_api
|__call__
*}&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 400}}}}
�
)	constants
*	variables
+trainable_variables
,regularization_losses
-	keras_api
~__call__
*&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Relu_9", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "Relu_9", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_9", "op": "Relu", "input": ["batch_normalization_16/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}}
�

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "dense_19", "trainable": true, "dtype": "float64", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}}
�
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:trainable_variables
;regularization_losses
<	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float64", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 300}}}}
�
=	constants
>	variables
?trainable_variables
@regularization_losses
A	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Relu_10", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "Relu_10", "trainable": true, "dtype": "float64", "node_def": {"name": "Relu_10", "op": "Relu", "input": ["batch_normalization_17/Identity"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {}}}
�

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "dense_20", "trainable": true, "dtype": "float64", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.003, "maxval": 0.003, "seed": null}}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.003, "maxval": 0.003, "seed": null}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}}
�
H	constants
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_3", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "Mul_3", "trainable": true, "dtype": "float64", "node_def": {"name": "Mul_3", "op": "Mul", "input": ["dense_20/Identity", "Mul_3/y"], "attr": {"T": {"type": "DT_DOUBLE"}}}, "constants": {"1": 1.0}}}
"
	optimizer
�
0
1
2
3
4
5
!6
"7
#8
$9
.10
/11
512
613
714
815
B16
C17"
trackable_list_wrapper
v
0
1
2
3
!4
"5
.6
/7
58
69
B10
C11"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Mlayers
	variables
Nlayer_regularization_losses
Ometrics
trainable_variables
Pnon_trainable_variables
regularization_losses
u__call__
w_default_save_signature
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
*:(2batch_normalization_15/gamma
):'2batch_normalization_15/beta
2:0 (2"batch_normalization_15/moving_mean
6:4 (2&batch_normalization_15/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Qlayers
	variables
Rlayer_regularization_losses
Smetrics
trainable_variables
Tnon_trainable_variables
regularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_18/kernel
:�2dense_18/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Ulayers
	variables
Vlayer_regularization_losses
Wmetrics
trainable_variables
Xnon_trainable_variables
regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_16/gamma
*:(�2batch_normalization_16/beta
3:1� (2"batch_normalization_16/moving_mean
7:5� (2&batch_normalization_16/moving_variance
<
!0
"1
#2
$3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Ylayers
%	variables
Zlayer_regularization_losses
[metrics
&trainable_variables
\non_trainable_variables
'regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
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

]layers
*	variables
^layer_regularization_losses
_metrics
+trainable_variables
`non_trainable_variables
,regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_19/kernel
:�2dense_19/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

alayers
0	variables
blayer_regularization_losses
cmetrics
1trainable_variables
dnon_trainable_variables
2regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_17/gamma
*:(�2batch_normalization_17/beta
3:1� (2"batch_normalization_17/moving_mean
7:5� (2&batch_normalization_17/moving_variance
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�

elayers
9	variables
flayer_regularization_losses
gmetrics
:trainable_variables
hnon_trainable_variables
;regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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

ilayers
>	variables
jlayer_regularization_losses
kmetrics
?trainable_variables
lnon_trainable_variables
@regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_20/kernel
:2dense_20/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

mlayers
D	variables
nlayer_regularization_losses
ometrics
Etrainable_variables
pnon_trainable_variables
Fregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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

qlayers
I	variables
rlayer_regularization_losses
smetrics
Jtrainable_variables
tnon_trainable_variables
Kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
#2
$3
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
#0
$1"
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
.
70
81"
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
�2�
+__inference_model_6_layer_call_fn_203566684
+__inference_model_6_layer_call_fn_203566738
+__inference_model_6_layer_call_fn_203567121
+__inference_model_6_layer_call_fn_203567144�
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
�2�
F__inference_model_6_layer_call_and_return_conditional_losses_203567098
F__inference_model_6_layer_call_and_return_conditional_losses_203566629
F__inference_model_6_layer_call_and_return_conditional_losses_203567014
F__inference_model_6_layer_call_and_return_conditional_losses_203566598�
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
�2�
$__inference__wrapped_model_203565973�
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
annotations� *'�$
"�
input_10���������
�2�
:__inference_batch_normalization_15_layer_call_fn_203567255
:__inference_batch_normalization_15_layer_call_fn_203567264�
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
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203567246
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203567223�
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
,__inference_dense_18_layer_call_fn_203567281�
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
G__inference_dense_18_layer_call_and_return_conditional_losses_203567274�
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
:__inference_batch_normalization_16_layer_call_fn_203567401
:__inference_batch_normalization_16_layer_call_fn_203567392�
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
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203567360
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203567383�
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
6__inference_tf_op_layer_Relu_9_layer_call_fn_203567411�
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
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_203567406�
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
,__inference_dense_19_layer_call_fn_203567428�
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
G__inference_dense_19_layer_call_and_return_conditional_losses_203567421�
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
:__inference_batch_normalization_17_layer_call_fn_203567539
:__inference_batch_normalization_17_layer_call_fn_203567548�
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
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203567530
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203567507�
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
7__inference_tf_op_layer_Relu_10_layer_call_fn_203567558�
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
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_203567553�
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
,__inference_dense_20_layer_call_fn_203567576�
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
G__inference_dense_20_layer_call_and_return_conditional_losses_203567569�
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
5__inference_tf_op_layer_Mul_3_layer_call_fn_203567587�
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
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_203567582�
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
7B5
'__inference_signature_wrapper_203566876input_10�
$__inference__wrapped_model_203565973�$!#"./8576BC1�.
'�$
"�
input_10���������
� "E�B
@
tf_op_layer_Mul_3+�(
tf_op_layer_Mul_3����������
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203567223b3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
U__inference_batch_normalization_15_layer_call_and_return_conditional_losses_203567246b3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
:__inference_batch_normalization_15_layer_call_fn_203567255U3�0
)�&
 �
inputs���������
p
� "�����������
:__inference_batch_normalization_15_layer_call_fn_203567264U3�0
)�&
 �
inputs���������
p 
� "�����������
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203567360d#$!"4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
U__inference_batch_normalization_16_layer_call_and_return_conditional_losses_203567383d$!#"4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
:__inference_batch_normalization_16_layer_call_fn_203567392W#$!"4�1
*�'
!�
inputs����������
p
� "������������
:__inference_batch_normalization_16_layer_call_fn_203567401W$!#"4�1
*�'
!�
inputs����������
p 
� "������������
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203567507d78564�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
U__inference_batch_normalization_17_layer_call_and_return_conditional_losses_203567530d85764�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
:__inference_batch_normalization_17_layer_call_fn_203567539W78564�1
*�'
!�
inputs����������
p
� "������������
:__inference_batch_normalization_17_layer_call_fn_203567548W85764�1
*�'
!�
inputs����������
p 
� "������������
G__inference_dense_18_layer_call_and_return_conditional_losses_203567274]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� �
,__inference_dense_18_layer_call_fn_203567281P/�,
%�"
 �
inputs���������
� "������������
G__inference_dense_19_layer_call_and_return_conditional_losses_203567421^./0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
,__inference_dense_19_layer_call_fn_203567428Q./0�-
&�#
!�
inputs����������
� "������������
G__inference_dense_20_layer_call_and_return_conditional_losses_203567569]BC0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
,__inference_dense_20_layer_call_fn_203567576PBC0�-
&�#
!�
inputs����������
� "�����������
F__inference_model_6_layer_call_and_return_conditional_losses_203566598v#$!"./7856BC9�6
/�,
"�
input_10���������
p

 
� "%�"
�
0���������
� �
F__inference_model_6_layer_call_and_return_conditional_losses_203566629v$!#"./8576BC9�6
/�,
"�
input_10���������
p 

 
� "%�"
�
0���������
� �
F__inference_model_6_layer_call_and_return_conditional_losses_203567014t#$!"./7856BC7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
F__inference_model_6_layer_call_and_return_conditional_losses_203567098t$!#"./8576BC7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
+__inference_model_6_layer_call_fn_203566684i#$!"./7856BC9�6
/�,
"�
input_10���������
p

 
� "�����������
+__inference_model_6_layer_call_fn_203566738i$!#"./8576BC9�6
/�,
"�
input_10���������
p 

 
� "�����������
+__inference_model_6_layer_call_fn_203567121g#$!"./7856BC7�4
-�*
 �
inputs���������
p

 
� "�����������
+__inference_model_6_layer_call_fn_203567144g$!#"./8576BC7�4
-�*
 �
inputs���������
p 

 
� "�����������
'__inference_signature_wrapper_203566876�$!#"./8576BC=�:
� 
3�0
.
input_10"�
input_10���������"E�B
@
tf_op_layer_Mul_3+�(
tf_op_layer_Mul_3����������
P__inference_tf_op_layer_Mul_3_layer_call_and_return_conditional_losses_203567582_6�3
,�)
'�$
"�
inputs/0���������
� "%�"
�
0���������
� �
5__inference_tf_op_layer_Mul_3_layer_call_fn_203567587R6�3
,�)
'�$
"�
inputs/0���������
� "�����������
R__inference_tf_op_layer_Relu_10_layer_call_and_return_conditional_losses_203567553a7�4
-�*
(�%
#� 
inputs/0����������
� "&�#
�
0����������
� �
7__inference_tf_op_layer_Relu_10_layer_call_fn_203567558T7�4
-�*
(�%
#� 
inputs/0����������
� "������������
Q__inference_tf_op_layer_Relu_9_layer_call_and_return_conditional_losses_203567406a7�4
-�*
(�%
#� 
inputs/0����������
� "&�#
�
0����������
� �
6__inference_tf_op_layer_Relu_9_layer_call_fn_203567411T7�4
-�*
(�%
#� 
inputs/0����������
� "�����������