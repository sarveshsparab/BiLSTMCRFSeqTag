ۉ
�5�5
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
�
	BlockLSTM
seq_len_max	
x"T
cs_prev"T
h_prev"T
w"T
wci"T
wcf"T
wco"T
b"T
i"T
cs"T
f"T
o"T
ci"T
co"T
h"T"
forget_biasfloat%  �?"
	cell_clipfloat%  @@"
use_peepholebool( "
Ttype:
2
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	
�
)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
�
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0���������"
value_indexint(0���������"+

vocab_sizeint���������(0���������"
	delimiterstring	�
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
�
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
2
LookupTableSizeV2
table_handle
size	�
!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
�
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
�
ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:�
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype�
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype�
9
TensorArraySizeV3

handle
flow_in
size�
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring �
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype�
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �"serve*1.13.12
b'unknown'8��


global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
shape: *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
z
wordsPlaceholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
a
nwordsPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
�
charsPlaceholder*2
shape):'���������������������������*
dtype0*=
_output_shapes+
):'���������������������������
{
ncharsPlaceholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
k
 string_to_index/hash_table/ConstConst*
valueB	 R
���������*
dtype0	*
_output_shapes
: 
�
string_to_index/hash_tableHashTableV2*8
shared_name)'hash_table_../dev/vocab.words.txt_-2_-1*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
�
4string_to_index/hash_table/table_init/asset_filepathConst*'
valueB B../dev/vocab.words.txt*
dtype0*
_output_shapes
: 
�
%string_to_index/hash_table/table_initInitializeTableFromTextFileV2string_to_index/hash_table4string_to_index/hash_table/table_init/asset_filepath*
	key_index���������*
value_index���������
m
"string_to_index_1/hash_table/ConstConst*
valueB	 R
���������*
dtype0	*
_output_shapes
: 
�
string_to_index_1/hash_tableHashTableV2*8
shared_name)'hash_table_../dev/vocab.chars.txt_-2_-1*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
�
6string_to_index_1/hash_table/table_init/asset_filepathConst*'
valueB B../dev/vocab.chars.txt*
dtype0*
_output_shapes
: 
�
'string_to_index_1/hash_table/table_initInitializeTableFromTextFileV2string_to_index_1/hash_table6string_to_index_1/hash_table/table_init/asset_filepath*
	key_index���������*
value_index���������
�
$string_to_index_1_Lookup/hash_bucketStringToHashBucketFastchars*
num_buckets*=
_output_shapes+
):'���������������������������
�
*string_to_index_1_Lookup/hash_table_LookupLookupTableFindV2string_to_index_1/hash_tablechars"string_to_index_1/hash_table/Const*

Tout0	*=
_output_shapes+
):'���������������������������*	
Tin0
s
(string_to_index_1_Lookup/hash_table_SizeLookupTableSizeV2string_to_index_1/hash_table*
_output_shapes
: 
�
string_to_index_1_Lookup/AddAdd$string_to_index_1_Lookup/hash_bucket(string_to_index_1_Lookup/hash_table_Size*
T0	*=
_output_shapes+
):'���������������������������
�
!string_to_index_1_Lookup/NotEqualNotEqual*string_to_index_1_Lookup/hash_table_Lookup"string_to_index_1/hash_table/Const*
T0	*=
_output_shapes+
):'���������������������������
�
string_to_index_1_LookupSelect!string_to_index_1_Lookup/NotEqual*string_to_index_1_Lookup/hash_table_Lookupstring_to_index_1_Lookup/Add*
T0	*=
_output_shapes+
):'���������������������������
�
1chars_embeddings/Initializer/random_uniform/shapeConst*#
_class
loc:@chars_embeddings*
valueB"$   d   *
dtype0*
_output_shapes
:
�
/chars_embeddings/Initializer/random_uniform/minConst*#
_class
loc:@chars_embeddings*
valueB
 *AW�*
dtype0*
_output_shapes
: 
�
/chars_embeddings/Initializer/random_uniform/maxConst*#
_class
loc:@chars_embeddings*
valueB
 *AW>*
dtype0*
_output_shapes
: 
�
9chars_embeddings/Initializer/random_uniform/RandomUniformRandomUniform1chars_embeddings/Initializer/random_uniform/shape*
T0*#
_class
loc:@chars_embeddings*
dtype0*
_output_shapes

:$d
�
/chars_embeddings/Initializer/random_uniform/subSub/chars_embeddings/Initializer/random_uniform/max/chars_embeddings/Initializer/random_uniform/min*
T0*#
_class
loc:@chars_embeddings*
_output_shapes
: 
�
/chars_embeddings/Initializer/random_uniform/mulMul9chars_embeddings/Initializer/random_uniform/RandomUniform/chars_embeddings/Initializer/random_uniform/sub*
T0*#
_class
loc:@chars_embeddings*
_output_shapes

:$d
�
+chars_embeddings/Initializer/random_uniformAdd/chars_embeddings/Initializer/random_uniform/mul/chars_embeddings/Initializer/random_uniform/min*
T0*#
_class
loc:@chars_embeddings*
_output_shapes

:$d
�
chars_embeddings
VariableV2*
shape
:$d*#
_class
loc:@chars_embeddings*
dtype0*
_output_shapes

:$d
�
chars_embeddings/AssignAssignchars_embeddings+chars_embeddings/Initializer/random_uniform*
T0*#
_class
loc:@chars_embeddings*
_output_shapes

:$d
�
chars_embeddings/readIdentitychars_embeddings*
T0*#
_class
loc:@chars_embeddings*
_output_shapes

:$d
|
embedding_lookup/axisConst*#
_class
loc:@chars_embeddings*
value	B : *
dtype0*
_output_shapes
: 
�
embedding_lookupGatherV2chars_embeddings/readstring_to_index_1_Lookupembedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*#
_class
loc:@chars_embeddings*A
_output_shapes/
-:+���������������������������d
�
embedding_lookup/IdentityIdentityembedding_lookup*
T0*A
_output_shapes/
-:+���������������������������d
�
dropout/IdentityIdentityembedding_lookup/Identity*
T0*A
_output_shapes/
-:+���������������������������d
c
SequenceMask/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
T
SequenceMask/MaxMaxncharsSequenceMask/Const*
T0*
_output_shapes
: 
V
SequenceMask/Const_1Const*
value	B : *
dtype0*
_output_shapes
: 
V
SequenceMask/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
~
SequenceMask/RangeRangeSequenceMask/Const_1SequenceMask/MaxSequenceMask/Const_2*#
_output_shapes
:���������
f
SequenceMask/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
SequenceMask/ExpandDims
ExpandDimsncharsSequenceMask/ExpandDims/dim*
T0*4
_output_shapes"
 :������������������
�
SequenceMask/CastCastSequenceMask/ExpandDims*

SrcT0*4
_output_shapes"
 :������������������*

DstT0
�
SequenceMask/LessLessSequenceMask/RangeSequenceMask/Cast*
T0*=
_output_shapes+
):'���������������������������
E
ShapeShapedropout/Identity*
T0*
_output_shapes
:
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
_
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_1StridedSliceShapestrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
K
mulMulstrided_slicestrided_slice_1*
T0*
_output_shapes
: 
h
strided_slice_2/stackConst*
valueB:
���������*
dtype0*
_output_shapes
:
j
strided_slice_2/stack_1Const*
valueB:
���������*
dtype0*
_output_shapes
:
a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_2StridedSliceShapestrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
Q
Reshape/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
j
Reshape/shapePackmulstrided_slice_2Reshape/shape/2*
T0*
N*
_output_shapes
:
s
ReshapeReshapeSequenceMask/LessReshape/shape*
T0
*4
_output_shapes"
 :������������������
f
ToFloatCastReshape*

SrcT0
*4
_output_shapes"
 :������������������*

DstT0
S
Reshape_1/shape/2Const*
value	B :d*
dtype0*
_output_shapes
: 
n
Reshape_1/shapePackmulstrided_slice_2Reshape_1/shape/2*
T0*
N*
_output_shapes
:
v
	Reshape_1Reshapedropout/IdentityReshape_1/shape*
T0*4
_output_shapes"
 :������������������d
_
mul_1Mul	Reshape_1ToFloat*
T0*4
_output_shapes"
 :������������������d
�
.conv1d/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@conv1d/kernel*!
valueB"   d   2   *
dtype0*
_output_shapes
:
�
,conv1d/kernel/Initializer/random_uniform/minConst* 
_class
loc:@conv1d/kernel*
valueB
 *�{�*
dtype0*
_output_shapes
: 
�
,conv1d/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@conv1d/kernel*
valueB
 *�{�=*
dtype0*
_output_shapes
: 
�
6conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv1d/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv1d/kernel*
dtype0*"
_output_shapes
:d2
�
,conv1d/kernel/Initializer/random_uniform/subSub,conv1d/kernel/Initializer/random_uniform/max,conv1d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1d/kernel*
_output_shapes
: 
�
,conv1d/kernel/Initializer/random_uniform/mulMul6conv1d/kernel/Initializer/random_uniform/RandomUniform,conv1d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:d2
�
(conv1d/kernel/Initializer/random_uniformAdd,conv1d/kernel/Initializer/random_uniform/mul,conv1d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:d2
�
conv1d/kernel
VariableV2*
shape:d2* 
_class
loc:@conv1d/kernel*
dtype0*"
_output_shapes
:d2
�
conv1d/kernel/AssignAssignconv1d/kernel(conv1d/kernel/Initializer/random_uniform*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:d2
|
conv1d/kernel/readIdentityconv1d/kernel*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:d2
�
conv1d/bias/Initializer/zerosConst*
_class
loc:@conv1d/bias*
valueB2*    *
dtype0*
_output_shapes
:2
s
conv1d/bias
VariableV2*
shape:2*
_class
loc:@conv1d/bias*
dtype0*
_output_shapes
:2
�
conv1d/bias/AssignAssignconv1d/biasconv1d/bias/Initializer/zeros*
T0*
_class
loc:@conv1d/bias*
_output_shapes
:2
n
conv1d/bias/readIdentityconv1d/bias*
T0*
_class
loc:@conv1d/bias*
_output_shapes
:2
^
conv1d/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
^
conv1d/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
conv1d/conv1d/ExpandDims
ExpandDimsmul_1conv1d/conv1d/ExpandDims/dim*
T0*8
_output_shapes&
$:"������������������d
`
conv1d/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
conv1d/conv1d/ExpandDims_1
ExpandDimsconv1d/kernel/readconv1d/conv1d/ExpandDims_1/dim*
T0*&
_output_shapes
:d2
�
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/ExpandDimsconv1d/conv1d/ExpandDims_1*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"������������������2
�
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d/Conv2D*
squeeze_dims
*
T0*4
_output_shapes"
 :������������������2
�
conv1d/BiasAddBiasAddconv1d/conv1d/Squeezeconv1d/bias/read*
T0*4
_output_shapes"
 :������������������2
d
mul_2Mulconv1d/BiasAddToFloat*
T0*4
_output_shapes"
 :������������������2
J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
subSubsub/xToFloat*
T0*4
_output_shapes"
 :������������������
`
Min/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
o
MinMinmul_2Min/reduction_indices*
	keep_dims(*
T0*+
_output_shapes
:���������2
U
mul_3MulsubMin*
T0*4
_output_shapes"
 :������������������2
W
addAddmul_2mul_3*
T0*4
_output_shapes"
 :������������������2
`
Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
X
MaxMaxaddMax/reduction_indices*
T0*'
_output_shapes
:���������2
_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_3StridedSliceShapestrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
_
strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_4StridedSliceShapestrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
S
Reshape_2/shape/2Const*
value	B :2*
dtype0*
_output_shapes
: 
z
Reshape_2/shapePackstrided_slice_3strided_slice_4Reshape_2/shape/2*
T0*
N*
_output_shapes
:
i
	Reshape_2ReshapeMaxReshape_2/shape*
T0*4
_output_shapes"
 :������������������2
�
"string_to_index_Lookup/hash_bucketStringToHashBucketFastwords*
num_buckets*0
_output_shapes
:������������������
�
(string_to_index_Lookup/hash_table_LookupLookupTableFindV2string_to_index/hash_tablewords string_to_index/hash_table/Const*

Tout0	*0
_output_shapes
:������������������*	
Tin0
o
&string_to_index_Lookup/hash_table_SizeLookupTableSizeV2string_to_index/hash_table*
_output_shapes
: 
�
string_to_index_Lookup/AddAdd"string_to_index_Lookup/hash_bucket&string_to_index_Lookup/hash_table_Size*
T0	*0
_output_shapes
:������������������
�
string_to_index_Lookup/NotEqualNotEqual(string_to_index_Lookup/hash_table_Lookup string_to_index/hash_table/Const*
T0	*0
_output_shapes
:������������������
�
string_to_index_LookupSelectstring_to_index_Lookup/NotEqual(string_to_index_Lookup/hash_table_Lookupstring_to_index_Lookup/Add*
T0	*0
_output_shapes
:������������������
��
Variable/initial_valueConst*��
value��B��	B�"���$�='��>;��ƊZ=�b">E�>>���["W���@�н��>V� ?|�
�� �
�I=?���Q�?�����+�>l�<��:�$����>�þ�}f��0ļ�䛾��k?�H�EGҾ_F�>��>�^5����\v���=��3>�"0��J�9�T��E>K卾���>�>��	?JA�m9����<
�Ѿ��=�'�=	'��7;�~R�=k+>���OX�F%�>8�"��m���}.� )���ו=̙��J{3��H"���ƽ冕����>@�:��A/>G�|=
�}=�M�>$ ?�|�>�!�=�FӾiR?y@Y�O��,��=�GC�,�E>M2r��E>l�D��d0?t��=���<�H���d��G�>���=��W>[_�>��=^����S,�:z\��>]��lֽ*�Y��?�u��D݇<�:����_�Hm>Fм�/����(?�:��e�>��|
@�:;��-�=�T$�IT>5慨�`>!<���:?l��>�p��~Rm�^����:�1��YY=�q�Q�`>���>y
�g���FF�n2��37��ݵ ���G>zVR=*:�|,��=']�.��N�>�#E�<��e6h����=b�����>S\u�6J<L��=Sy��%��Ɣ� $�'NN>��H������Ԃ׽��>���g<>�:#J��C��0�=�	��^������Dr�g,�����=��>Έ��%;������bB���a>��
<U1U���>�yF�;�<�٤�d(�?5�����֭�>ȵ�>(a&>Z������(��,L>ގ��T ,����<|�~�/%�;@��=���� >���33���s7>�r����<Á ?�ܿ�h\�=mŎ�Ou��U�5|�=\�N>��>='�=����˜?��
>����H"�vTž��x��w�;���$|�;gaO�����i�!��G�?�K�>n�ͽ�2�>�'�>�����x;��l=��ּ5A�=��!?���T��p�=;|>s�>�`Q�� G�F�ؾ_��N�>V�=i�?���=]mE<!�=�f�=��->�i>ޫ�>Z ������V����1���� �<T��<O6>\U���8�<��(>�p>n����+��XVھ��h�����O>��ξ��*��0�>σ��>=6<j���P�<p5>J)(>
�h���l����:���<� �R~r>�ؙ����P�>�������-=g{t�<�@{�>�;>AJ>|����K�8�̽�
U��e^?@���~WD�ǡ~=ȳ�����=��=>Է��'����d�1��Y ?�S��lm�:�]>h?r>3(d;� a>/O�=��=���>RI=>H���w�Խ,�>����P�W>UM0>��>�|��+����<p5���_=<I��N{ʽN(D>�:���aÓ�>���=)�!>�1��Wо��=1�S>����y#3��/,>k�>(�R>b�<It>ɓ?t$׽,��=Sx�̗�>;�?��	�=��
?D�=!���l�%�~=�u���X��.1�JF��vl?. t=t^���5>�A��2�>�m7��:
>Ҝ��EϾ�þ=�鎽U0�K>L7)����0��>=a�m>�����C=z��>�'����t9?TF����>&��=���=D�a>r�Ӿ� ܼg�=Ttd�>u�<�ֳ�u�?ỳ���=����E�޾��v� �ٽ�p�uv�"qϽ��M>��&��Ԇ�Nё�����*�<~�� j>�`�½毐�����eߕ=;6�>�Y�<��P�hF=�߮�$'S�~��<!��)�>����5M�0�a� W��־�">�L���	.�@ὰ8��:�?��뾘 >����; �=Ɖ�=S�P>��;R�����d#p�
��=��=�-�=��=�)��%����.�Mj�=YM�=���><1�G<�<y�=�T�>�+<$�>�씾mƉ=qL>��=��=`��<*��>`�$>��t��_0>�a�<u�=��K��F
�v��.?�����dP>�ѾS��=b�۾��6>� �>
�<t)�>����?�� �$�0��>%u��0�<�1�>0���9���[=�齥,�n�'�=���*�9>�Q�mow�
�i>�7>EV�V�f��\�>t$w>$� =�j>yu�>������=0��Dݽ�`���D�>�ʘ�=D#�(CU=�s=U��>[�y�ӇN�9�_>B�?L��<�Y"?���=6׽�S`>28*>�#�=*�>0�?�J�=(~ܾz�U�2�d��S���Z�&9>��i�q��=�z�=g,>��P�������}<4=/���<ֺ�4�
j8=�[�>�z��yxO�W����fr��'�=�B
>Έ��VH9���+>�E��s��)��>�O>*t>>
���>�6>� ���;����$@ep��4)�:�=i)<�S���<͑��fk?��� ���нٵ}=�ՕϾ��>���=�
=h�j>��>��:�κ���>��]�pB!>>!{=�ᙾ5��>���>�-=󫙾��=���="oy=(��[ӌ>8�
�vO^�~:�<r=='��G ��g�� ����O�>ˡE>�QF=��Q����=�^�=�+¾8�==;�ضh���>^��>2Z'����"qO>��C�L+��{�=e�>��_>|a�=�F��=&$=By=1�ڹ&�>r�=û�>�`�>�.�=��n>��O�_FA��	\�Z��=i�����.��N�>û<�?�оM�>��<>fI@=q=*�7��>�u�=��>{�#<�7��[�*��q$��:*?��>8}=��k�=E���$>k� ?�5w�&��;v�н"�>>f1�=����g3>K<�>Mi}=�oo��,L>~>���,�z>�i�ϟ6�����쾾5)��H��{�>�H����X8�=�+e>#� ?��Ծ���R���;�=�6�=:#�>�4���n;��S㽢ѭ>j�^-w������X�>��=�>>�⇽��׾"0��^)>�}#�⬨�E����>�*!�~Ś=$d������1�>;�5�}y?Ԝ�<,����o�-x��ǻ���`�T:8��?_�]�__+�=,�>m�=X�:<��
��m��s�>{1�>���<~t����þ)yվc
�=L�ƾ4�=6v)��,	>�M>E*������(v>i㘾��'��8<r�#=��f>5$n�?��%�ν"T�dھ�v�k+־v6�=ٙҾX��>ބ>_��Z���V}.������Z����>V}.�&��Z���\��Vס����=D3�=���>7ࣾ"�=�59�2��=ÁP�� D�>{���I�����=I._��\��֋	?Y5<xbv>�c9=Dn�����}?�=�νgDi�Kȇ��j���w�>��$�:$���>ɰ
����<Y�>@�>p����^O>�:O���>���B��=�?��`Ⱥ�%]�>J���Ae�>}�!=��m�ϾbJ>b�=�����<�/�>ı.>q>�t�>�뒽Hm�=�8��[#>.H���e>ؼʽ�.��K�>G�>�`>0���p_'>����X.=*½�ړ>�?��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ��V�,��T5��jj>�����վ}�G�(�y=�Vm=?RL?���9��<��%>0��>�{,��<��ǲ�B>�?"q�q�q>8 ���8澑�����>�M>H����_�=�,ҼS�=��(C��{�&��x�>�[i�~0?fNW=��2?���y#S��$��e,?�<?��?h��w2>�s߾�풾��ݾ/���>����!�>X��=r�>4H��ٙ>Lq�>C7{��&>�����=w�?Z�>4.\>!"���9]�oy��a��>eފ�NN?�玽䃞�����f�E�0�,Ժ��1
=��;Лھ�������B�{�
�t>qɡ��q�N�W��9v�e�#�֨G>~r�����<U�?�6��"��̗��(
�>C��<�>�ު>��=s�ƾm9�=<Ž����#�;p�?Sy�=ß��N(����&?���>�>=�U>�aU��>��ʾ��A>"8�<�M�>N��=�׽��,�b�H��Ѱ>�
�=Q�)>�4l>"�H��m�>�=>M/�<�y>�U(>�Q�s�>𿕾���k}�>����.���>���=�>W>�rN��{��]K��C(<Z�	?i:��y#s=q?�ݟ����>���"q�>1|�%@�>Թ�<����`��X瘽|a�>`�>�:�;;�Z�s����;:�0>Է,>D�g>_	����>�T�1|�=��¾�4�>�gv>V�z>�X��V�����N��3�>!�^�����#-��Bxľɫ�>=']�\"�4�F�s>��1��#>m���no��wվ,*��:�&6?��v�=b�<>�m����y0:8�5�=:佲��<���-�ݽ�\�>D�	����=�[}>1�>
��� ���|=����s�>f��]�=�p�>F��<ɻ��^��5�?[���q.=��>)"?+#=�B��4�N�1���>�}��M���T�>�B�;�I��g�>t$Ǿ�����>��B>#�ؾj�>�׹�s4�1�پy�>,+�>��<��e>o�>R� ��(>��=4���>�>���>%�>�+>)�g��s?���>�n�>�ȾVHپ�|6�&3�(
�W%��?�b�P��rн���>.s�>W�>�HB>���Kz>g����D�����d�t�0=�=�����N(���	��9K���W>Xs�>��>ۅ�&�>:�;�=`��I����_M�ԂW>�]�K�d�Ll6?�n�=�ֻ��@^�?��=`̾k0>�~�걍=T�y>�ix�s���� A>"r�=���<tA���h�<�"�;��O=�Kw>��>�b���=hB�=W>�<LS=|
 �_
��o-�<�U����=���>�X�}\>U۽Zg���l&�{1Խ6v)��w"�^c7>���=ׅ_=AQ���g<�����ɾ��E>('��M�>,��=Qfþ�]=�[7>B�����>�Õ�ga�>&S��j�:=�쳽�i>�N�=wJ��#׍=	3?�D�=� ��$(ξ���>����
�=����S#���>"�?>��	�>m�x���$>�Y�aq���&��K�>:����p�l>�9b�>�о���^�߾F��Q� �������׾�@���b��Y�<�\�?Nb>�y�+�:?��O=X����H���_->3m>v7��Q�>K�>*�������?\Z�� ���fN7>.�=~W4?#�>ɰ��Z/&��aԻc��>h��>h<��=>y�>�,�� ��>�q�>TR��S�>m�4�I._�b�3=�S�<7q�>*o�>�f�>��>A�C<���!ȁ>m9w��׈>����e=���=�=��|<|���nP�=w��=��(�K��Ou���g�6��=��>du�Υ8�H�>Ll��I>��>?帽�7A>�$پ�p�r3ܾoH#���Ծ�\���� =��W;���=�`�* �>T:�>�S#>:@�^׏�ں�;=7q��B>���tk9bJ�>8�=%z<����&���Ub��`t�˜��;6���\'>g~���k��>���?���;
�:>�,�>���	3M�tA]>('�>��ۼ���Zv>a\����>%���S� ?��=�'�=O#�>}�����>9+�<���>Զ�=_&
=� �=1�5=�_��TG>e�>6W=���D�>T䐽�M>!�>���؁�>!`�Q�����Z�J�=�O�>��(>A����b��Eؠ���f/=�sU>��� ��/�>J�;����>:��p�r�c>�8̾�C��� �=�����>_$�>�Y�=�;��&��\ �rĚ>�ȯ�[B�ض�A}������5{��mtμ�$龮ӈ�W[Q�穾��ޖ>�/�a&>��F>%!={�&?�M����ɾ+5�=[^9����������w>�l=��\>�u�A}��^r>rm�>!>0<{*@Q�&?�]>�����=d�+5�����>k�=>8�b�.�q=�B��k����p=���=g�����=K�M���˅J<��\��`Խ�d@;�� ����=ZGU>�@��L��"ݽ-!߾Ԃ׾�z�>%�< ���y�<"ỽ���9�t�=O��>(��<���=�zt>�н�e�����2>L<�?����>,�N�H3>��=Ef.=<�X����>��[>�ȏ>KY����>��� AQ>f�l>�����w9��>�(>?�о�=�����ֺm�>=='��w�=#-U� e= }S=� ��3+��v���HξP�I���Z>�K)=��>g�q�6Y#>��=�@վɗ=K�X>�����<����7h�{X����燾X�@e�[�c�E<*�%<�h��؞Y>���=?R���_�j��<�+���^�*����>��?|A���->^�U�C�C�����%�<�6�o>Z�#1�����?�!�=6<���)��=�ge=w+��5��=����">��J�=-`¾�Ϗ>ޫ����b>6�d�-x��u��� �>w�>�û�s� >�(��fǼ �g��(>����V>�_=�W>�T'>��5= *����q��#�����;cb��z��/��=�L>('Z�Ih�B\��
�>�O�=A��Y>$�<>L�>���=�F�=/��>�w>$^�<��>�Bнz�ݽ$�>�>>�=n�>���28
>8�J�A��=e�U�7��>n�󽦸��@��ۢ��!v�>�z0�)���>�XB<mw=���=|��>�������=��>��Y���Q����>7��� W�=��u=���=��=��=���B&�>Vν�d������mt��$	>��>�	>�G[�iW����!�W�>��ν�V=�C>�g��>'��=-`B>?�>�Zy>9>q�1>���w
>e�i>v��>#�>�쾚��<�3>�i��|� ���>���=�C>r���8�=af�$��>�!>R'�=9�>.=��i>�i�>yu>���`��=�&�>�}�`��;�I��(��U���5F+��R�=3��Q>S�x�=>V�\>��JM���E>ގP�b��<6��>F�>Qڽ'1�>a��=nQ��|<|���G)��P]�'�F>itg>Zྭ4����U�>>����g=���X�@�x����>R�2?�/�0���u��=������N?����Cn>�*�=|_�<+�+��`�>�!ԾGt�FB������?�>Ԃ׾ٔ�Ӈ�>�=rk�<��A��=ۿR>�������:�!=q���4L>V��<�*;g��>���>�&¾:�=���>��V���w<=e����=�Y>F�Y>�0ӽ,�ʽL����<DQ ���ȽA�޾#j�;��>? 	����j�W>�z��u:�<U��>�c�=('z>/�d>��>ʉ�>���>��:>��%@;Q��>�л�b���F�<�D>�[%�>5cQ��L��=~_��[.?�>�}#��>�y��)�>)?I>�j(>�y�>)�;>��=$b*> &��]�=D�=��R���(>����i��Ή���)�R��=֑=fk�t�?Y����y�=�Y��u�
�#��_�=��e��	�>To��轱=����/?o�J>���B��*U�kbA=�2$>��=6v!�ı�>��e>�#���<����=��&�[B>���b-�>�C�=��<�=a)�5�*��HQ>
�#�0����0���Z�=��r���H>���3���)��e����=4���W�5�X>�k���'�C�ž�3Z�n >�g�j��=oG�=%/�����c@>����dE�9�Y�8N�<��>��f��Y��j����V�����=�=�b���z����=�+ʽvp�=Xc�O l=O@��i:��(
4>��=�km��qǾ����$^=	� ���ؽ������pA�=lCe>+5�=@����Oa>��t�]��=���p���S㽧��=.�>�ʚ�ep��  ��v�;CV�=�2��q->�o>2qk=iW�=?o
?��`>ض����t<����E>�/=�>僧=`_��@?>qɱ��!E�t��>���=����T'>��'>�s5�u�l��<2������k+?^���ض�'��J>��=г�aQ��cbS��r�=��=>0�|>xC�=���N�<1�V>���>q̲<�^>	J>U��=Q7=�oy�؞��0���:P�t��=�Ǽ�3˾��<�՝>L�d>����z�=��J�z��?��J)�>�M�;&����>�F� �9>���=�;>h�>��[��=�S�=���>qZ�=����\ �q�E��Py=�+>��6�A���>�ý�0?)��P�|=�61>V���k��=#�?	�@��?��v��R�꽮�n�d�= �>����m�>��=t�[>~�8;�"������$�>��=�� �Uj�>I+�<�2>s�L>��/��V��՝�'�������7����,>X���*����=f���2>Օo>�$�>�G`=h�0�Y�*>` ��u��s���s��x]�]R�=� ��T�O�O�T��C.�>�+|���L>;R�=��{>aÓ>A��%]�>���=_^ �����xE0=��U������ս%��=1��=��=�?�=�6)������5�ۅ�=�I>�*>F���[z�=-[��D�R��QU�9�9>l���W����>�+̾�X�X��>.���ff&>j�Z>�X�??W�>�6D=g���rP"�����{�1?8	��N#<_�>����L���=���wȽg���:�@�߾�NC>A�=���[�_>+0ĽӀ��E��S���y���1}�ö>I��>�]>@�=�����=�y�=q=��:�=8٦=@js�eǆ������=Y�>,����(�=o/�>4hh�Mj輾M?�pB>��>5z�<ۧ#�d���-x��'f]��o��k�>��8��M]=:��>�P=�ͪ<�N��O�o=��O��4�=}�㾊]����<)}>h��>E�
>��A>�A��Yy=�;��=B�:>���=wʼJA��5z��Yir>�7A>��(���J�M��l�h=���>����G��=홅=f�C>cE;��ͼ$�=ŏ1��!����=��z��=!<ڽ���]���
�~�{�6��]���	�>�,ƾ�_P>�Y�v�ʾ��w>dɾC9����>:]v>���!>`��>X7޻�x�B>�28�>~Rm�ʌw=E/C�Ά!��y�>S���i�>F�?`�G��=�>��ý�,=��S=�K�=��b=����R�>G�<;|�������W�9�,� ?�6�>�^F>�G�=
��>�o���<	�_>k�>%X�>
�Z���+>+j>��>k�h>��	��>Ic4�ů���;�kݼv��>\��<��D�7�m>峜���$> �'�7l>1|�������ԾJ�[=)u�="�/�)"C��H�>ط>A�X�cA���t/<K�R>�Vt=��>Ԃ>������>t)�>�Q%?)�𾳘 ����Ic���=|>6��?�F��Om�����U�;p^�Ș�>V��A3c;��>Ș�>c;�]�>�� =6����>Hm">��'��S�>G>p���2=	�T&��	�2�(>B&!?���>�:�����>@x>��??�2?�x0=�s>�-������G���N]��N>�9:���=!�Ⱦ�ē�[�������4���7[�Kȇ<�K>8�뽎�=��8��yu>�&�>��H>(D >ף���m�X�=NB�<�:A>y��VD=$����rѾ3�=)�0>.�!>�"�=�&�>��>߉�>���>H���2w��W|C=/}�M���\}���$>!��>�&>ι>=#������P��T����^>�������>5���
>ۢl>���=F���u >Gr@s� �BC�>/i�>и�=X��#-�>��k>�dƼ�l�=�-�>�>w>@���a��'U>�^¾[(Y=J)>�|�&����Gg�9D=^�<q�羌��Zؓ�篾��>>F�\���>eS��Z����;��(�e==*>��R�sh�>�r=U���>z�ýd�7>K�8>�"���]>o�?���><��>�s��*W���!n���>��=�ő>��>*o���ǝ���z>�����e�>j��<��9>���y��=mŞ>"���Ԝ��4j=���;��C>`?�[�>p��>�:�>����bS�;b?�-�>\,�ܡ�=�I"?�0B><���{�6��V�����<��8���Z����;�jB�C��>�c��9@�t�g,:>h�,?M��(���޾�^/�rm(>�������S�;�P���?�>ܺ[>ۿ��du��-�;Ou>zS)���;��ľ4��>��ܾ�s�>�᳾�D�>	�@>���=�j�>H�>~ᕽ�`�>�c�>G��<v�>Xū�����	4X�('�=�N�>�y>+0$�]����Y��pL;qr�>�&n>�;��t�=n:�S�a>���r�پ�	'�p�>���>�GG�*��>�k=�S����?�T����K��g�>{�?�zg��Ö<@���g,�>�!�>=Dc�aP�=[�_>,=��=r��<-��;���8�=���>���=o��=�;��x-?p��=�,O����w���G�>⬨=Z/־	��@0���A�VN�Ƨ0�;��7�MJ!>Bx<?�>׽�پ�mѼQ��=H����E<=!v�>"�&p�1��?}̽�̌���>�z>��=�y�>��W>@�e>��:��C���ʾ����~��IH�=U�t�j��>�@�=0A�˺=i�b���m>sc�e�=��T�Z˽��>dɼ�n=��F>�m�=4��@�=��A}+>2ZG�2)>l���
�#�ꑦ�Z�]��F�>����9�y>��?���3Ŝ=�5>.9�g��=z����d>���5Y�>b����(��W	ƾ�����暾��ھ��>n�M<�.�=��S�2s��7 �<.V�����>*k=�*?�ޗ��]�샾��L=I��>�h���[�>�j�>�)���
��\D>k�?]P?>�G�=��{�_>D[��OX�>l�>���>M�>��������U���<<@c�>V�K��⍾��r�?�>�k2����>)?i>;�����q����2?�w ?Ŭ�>	��>���c)���V>�����;�����>�����k$=�H��d�>M�Ӿ/��>a�3>�{>����Ih�>�GξFә=���0���'�;MJ!>�ǜ����=�4C>8�>����3-?�׾0P��d!�A�c�AEU�k�=Z ��Z(=�ӽv#>#�->ę���=�6�>/���#�?ţ�<fཹ��ԇ>��Nz�= �>�瑾��9>[����K�=C�=s�
����>�	<�վG�+�(~L>EG
?� >�k�>5���j�>S�J>l"3=��l���=`ݾ��K��<��m�=s�������죳���۾�\>�������> |>�y��$(��	�=/�x>�=����Y>C>�^ �X������]n��~�=�Ű>B
�y#S��4I��n?ٮ��g��"�>�T����=G�M?Gw���½��>���=rm�>��/>}¾|�>p%;������Uޮ����;%t��>�>Di�>vC>�䥾"l������8�>4�ڽ^���
���>�=��C�������e��>2 ������ �b��>e����x�=X9��[��M�?���ff>���&�&�a�M��u�=,J�=�Ͼ��>���>�\5?���iT�<����w?f��>&�r�6鶽���ͯ&���> >"O�gDA?Υ8>帣�ze>Y�d>�ݽ���ۧc>d9�a���.Y�u���I���@�C�T=_AZ>H�t���E=�0�>P5z=��w�+��?��*>�\=y��:��|���f���ƽ�zk�D�=�Ȍ�+5?�,ϼ�s�>��)>K��>J�{��d��OǼb�K>�o��߮��B�>uͤ>������>S�>�U�<�!�=�b0�q=b��{?���=�1E=�<�<yuξep�2 ���z�<��!>G�>�!>	���=�>�R�=,H�>(e��d;=O�@>�PH>�\>>A=N���TԾo����-��%Ǿ�a>�H�F�v>.r=���>�X���v�=��
>��þ�cF>�4�������>$����q�=��r=/�>�w���y�=.����ׇ=��T⼙�0�u[�=�(�>���=nL�=�Y^=�r�od>>q�>��ý,H�հ_���4�p]1=���$�>���?�����ZY�:��>2� ��c*�>�>ף0�P5�=xN>2 +�hB����վ+�f?-�=t(ý�	a�j��>�ƾ{��>
JQ�1�'?t��P6%>�/h=/i����ؽ�㢽k�C����=_^��x��=��^����=.��E)a�L�V=�G��ݰ���/�� >�1@>������>���>�(<��Z��n�{�7Í��g>�Ѫ>=I:����=��ھj�?�'ڵ>V.>G�>3m_�� �x��F�=�b�p>��NE���>��=>]ݱ=���o)��Է,>h�=�$��>Q>[��0gv��E��>N�>�����[�e��󅽗��>�by��= �D>���i��?�g=&⭽���>(b= �=E���d��? ?v�¾��޾�ɽ>p���?g>+�>��>~ �XV�=sHj�q��=��v�p��>�+_>����߇��+�1>E*��(?��3>�^i>�X��i���;z�ܾ[R>}��=:]��c`��P�<�/��u<�=X9��>q�=.��.9�2=�>��>�>�������=�������	���r!�됋>�{X�Q�=��'�Gv���>^;M�>�gP���a:[��>I�Q�>"F�����E��j��:̗=P߂>Q̼�x���"��Ѯ¾�+C��W���F<ʈ�=�p:>2vB=�j��X�;�N��>T��^��=��>E���(~��1�=㥛����>�����?�YP?���>�{�>���>勼�>��{,>�G��5x��s>�gپ"�<i�<$���=��]n���D�*�->�!	���@��=�d�=���>���=����oy>Y�j= �L?W!��*��=��>�</�~>[��>� ��2<MH���R�<��<G=��׆*���T>�Z<���I���>��
> ��>��>M2>�!>��;i O>�c˽�BJ��4�>�������Tμ>\>U[>��;ܛ�=F|>�����>�;t���ؽ�GQ�G�'�xbv>|�>�ϟ���>�r>��ʾ��>��)>�s;p��>5�1=��=y#�>�^�>"��=��6=���g��=�Yҽ3�=��9=�3��*ች��'�>"&��w�儾�Qu�zV�m�{>�� =�S-���]�Ƚ���>fNW>���>F|>B>���C��>��ɵ���>�<t{I�5�½d;���Ȗ>[Ӭ����R��>D�=@����K�>�&e��ڭ>�L5>
�W>�,� A��ڏ�=���=�D�"����s�����QW>�1#�:X?����6Y�>��=� ���Js>`�>,+�=?�p>��}	�=F�i��u����m>��>(��|�>�=<��?��G<=�&������=��>�y��~��<��Ͻ���>��N�NbP��+ٽ�D���-^=ő���[�=���0/���=��>�9��75����g>9��=�)=��>;����i�=��&���=?�>
>J{��ۅ�=wH1=[Bξ����,eY�Ve?�����<�j�>��>:�>� �0؍=�O��솾�t�>|~�=f�>+���B�f>� �>P���8��>1���>YQ>��ѽ�1��Y�>��*��"U>r� >^ľ�b�Y,j;��
��*|<r�<�V��Gr��w�3>S����>�y�>/�<<jL=>S�����>��K��Fw��y�=Y�����Ľ�ѓ��<>� �=��v>}\K�U�h��$�>:��̸�=��c=�4>F;�!˂�KY澑,��|���P�>#�'��<%O��Lս�	�<�=!ʗ�4�"�G摽�>��X>��=�Q��Q���">�!�>	3�>�C7>OXb>���> ���&�����F��L�p�>\2�31�e�E=�2*>�z�>__�<�O>L�<�L8��-s��'I�\�<�t���� ��>s�����=,����[f>Ez��	+>jJ>��߽�e1?���T��(I�������>�.��n4�>��'�Ӄ�=����3��>P��>ض(?��?������>2Z�>-�>����#���<�>]�>�H�>'����P�F�P�<�NC>p�;38?���^�;��<��������A�iR
=d#0>��<�V�>Ra����=������ؽ�x@��Hp����pZ��)&�cb�=>�=Dn�>Ӽ�>���>�~�>���>��=��>�߾�9?�6z�B����6�>]6z=�p>1��vl��>���w�>�Y�>�K�=�0f��H�>R�ӽ�[���e7�r���¾/4�>�O�>m���2=�>��K=ף��q��=���%!?��Za���Y>�Vl�c+�<���>�]��5ӝ=/��>?O>���͖���;(~���[	>(~���
?}�3>6Yc?�E_�33;���)��!�Ҿ:����F����>U��>r����žg���ᓾ��>.?U�"��MH�����Fe=�fG=x�>�vս��iW�� J�iɾ��T�>����rE>��X>s�%�"��eQ>��o=�Ŀ}�'>cbs>w���[|*��>P�X��u��1F��+>Uq=�!<�>I*S=�_�>�w�>w?5��fL��H���ɾ���=ٱ��¾=Nz'�PS�>y�;<UM�>od�L�?�t��5)?���X績����/�S��žt�ʾZ��=B[�:]����J����r���>�ɮ�� �=t�o����>rV>��=�gH�`�>��ؽ���������A���H>�����������<�����=��Ǿ���>�ۡ>�+�>���,�T��پ!��>�x)=R� ? SF=J�۽3�> F��̃=/�¾͑<�>4׹>����I�>��n��z�>XV��bJ��L�>�'�=��!�^���H���RI%?�K���B��g,?��=���>�3?�侲�/����ʉ�>Tt>[�=d;?�E3���	����0��4���>����ٱ>�� ??:�ff&�A+P�ףP>~a>�xC�yR�O�޾77���Ǿ	��>�ѝ���:侬�R=	���w>?�1�¡7<C>��ڽ+�6>'f]>d'�O@�
����N�cz">����Y?�@>��*=�ž�_ӽt{�>:�+>^�4>���=%�>�mt�� j�^�=��`���X��?sV�`�g<��S��r�	�}?�V=>�?�2g>1_���_�I�'<�bἡ��?
����D�>:]��q>����=
>�X�.H�VX=�1>u���^5��R|�2w�>�&�=ff�o�λ�#>�z�����>�������M��>N�#=ʇ�=(�R��<�>�9>a���U�>�]�>��>�ȏ=0�>Y����q��1�>�K>��Ƚ��
��d>$��e�*��au��6�eSN�y��2r��g�2���?���=;>�4�=�]�>�����^�>��>����A}�<��>>�n�>��Ӿ�4�<�{�<i����Օ߾S��=�I�j�G?iuͻ	�q� o�=���F%?�Hb>��#�5bƽ�S;w��=��%>+�7�<ڈ�!d=�J���>�hh?��̽��e6��=֐8>w-�= x��rQ��v?h"L>ǡ>=zp��!�A>�Ɣ��롾�r��&�������՗%<-x��~�;���h����T=����E>���>���>�0�>Q�>VG�>�Ĝ=2�N>�̄��	?���>r�C>��
���>-	оA��<;�2��B��U�¸1�0��ܢ�S%�=+M
>Hp�K��>u�Ǿ��7>
�����EѾꕲ<5A4�W&�<kw���"��Z���⨼*��=e��=��>��]�XV�������e>� V>��r0>oӏ��&�>��f�&���?���&�����v�>��I��l>�=E�='��>����I�[�e߅���ӽ��z��K]>Lq5?�<v������|���7l>��>�ȩ>�S��VH9>�m#��w���i����? ��5Ң>NE�H0�>�J���6Խۊ��4�w>�5�>3��=����g�v�|�>�k�=yXؾ��[�n2*�Ae�=Lq5�`�=���z�}>��>�o��a7�><��>��=�ؙ>�>Ƚ�;Mu>�'�k�a�k+���"�U�*?Ƣi����=#2��7��!��v7�>E}����<W=4>?#��=�1 >�<���z��2�jlo�Y��>��=P�?���>@0G>#��=�p���:�F�ͽa�>��>R������˄��?޽��w�`>A���'��G2�;�m7��ڒ=٭�O]�>�j�Y7>\��=�m�>�Ն>`�>g�&<&pK>��>܀��d�={�oFM�^�u=J�N�A}k>�yd�m9��������>1B@?[�����t�����e=�2G�ףྌJҿ�Ƀ�**�I+��V9#�|�>���=�\*?�<�^���E������W�S�=G�E�6\����=��>Q����3?0�2�<�����)Z>T�e��=��]3پr�3�]�gD��=��MJA���Ͼk��>v�= �o>�bҾ��<��Ze�=���z�>A�=#��>ۢ�����=4��>�˟>�>��>��Q>[����U����>�ɪ�죣>�þTt��0��=&�����>-�?6YC��l���>'>�m�>>��>�2�=9�.>2UH�cK�l��=-!����>RE�=g�->���=�vu���=�6������w��=Ց>{���C�x�ؚ-���Ͼ�|?)�[��j'��8�>9(�����>�j>\l>��>,+m>���=
w�aã��w�>|~8>M2�>j�`����>P\i��I�>�F�p뾾^.��h>Dn�>!v�>��>�;�<��U������>�؅�b�<���=Ԛ�n>�� �r��h�<�U�>s��>#2,>�k��ix>����kl
<���{�>����c�.�\�s�`��	B��V�?Kv�>�`>`xe=�}����>�*�=3����b��$���5�NG@�B|�<x�H�Oy�<�ր��%��@K�����r.>�<)�"�ʾ ���� >N�c�:��=�������>����<�g�������ž�П>S"��*�M>B�½i��>Y1�=���KY��/�;�>vlD���¼�5�>[Υ>W!�>!�P?.s��9�����(=����*ľ��?�c+?�#?��>u>m�>ai<4�>��h>Ae|��Zӽ����b�@��X�+0�>�ă���"���>1�J���>ࡈ>JA��ۊ�>�>���>���dK<YiҽϠ����>�o<���t1��[?A����l�2�=�w�>z��%�h���꾩�?��?��>��H���->.�>[�ֽ��x>�:>��>���>Sy�ŏQ>��m>���=���>JF�>��/�yv�U��>5�&���uv
�C�^>�5𽟏��<W>����Fڽ��ѽ�} �=~7?�Bþ��>�8?j��>�!�C�>��>��=>lx�>`������WC⾈e3=T��<eV�ı�>�j�>����OU�S���=X@܀o�T ��K�d��#]�����X1�ƊZ=�?6�D������ ���$���˾V�>�S>��=��b=���<��>��q�B
=ё�=a2�=�g���w�>��[��2�>q����;��V���>\�?]�=>ݞ�\�4>�,���=�������ݽ�
r>֐��?��I>�m�=v��=mV�>&�>���>B&y>����� �+�?e6H���=Ra�>$����G�rl==�8��2�>��>^K�=
�����h=��0�+�n��A>��������c��>&J���A}+�K{�9,���]�=w�ž���<)>�Lu�����R�>9�<��C�/��Ƣ�>)]z� '<��>�j��䃎��y'>|w��/[�}�\�:L�9��>\�=A��=����QR>�9>Yک<���=�9�<�>*�Y�tA�>��>@��>)"C>���v齥���O��f;>Л*�|Պ��C�:�ْ=8J>�G�=�h#��!�>�\�;*X!�먊��߷��g���7���)>l�R>4.>�9=ZGž����zK��� ?I��������h=��>}�B>�,F�d�>[�E��j>)��=г���=�o����l=x5<��1��Љ�M�E�)"c>�>���%��=f�al�>!v�=s�������Ƕ��!z� )�uv�������܁�M2
�ͬ%��4�9��ƽ_A�>�Z��3Ҿ�b���	�����M>x�K>���=��y���=� ����	P�=�>o�վl*����>��=��>�$1<I�W=��@>����5P>5)��ʦ|�LOX>�J? �ԾGr�>5�=���V�+����֋��V�>a�=(~̾��=K�>� ��|ﯽ�Cѽy�>`?��>����ff����K��6>`����a�>巨=�S>��>��[%>9�ڼ�u�>_Aʾ������ca��V԰�>�==���>QMɼ9�>��>�v��;���=����?���> 8�����>
h"��Zs>$ͻC>J�o�Q��>�ﳽ�?!�	���)@�2!ƽLT��.y��!�>d�>R��>� d>��2��&��TƟ�G ��@��T���=
>y�%��ӽ}l\;��^>��P�D�=ĳ�=�z�\��W��R�>�Ao���:?P6��*:�=��=,�>�e񽯔e>��>E� @Ŭw�2����[=u�S�4����ҽ(�bg"?V�־���U�>W`H���a<��;=q >B�l��*B�/>�h>A�����=Lð>��=���=���>II���)��N(�>�7輻�>,�o= �C>3���0c���>���>���_$4�Q1�=$�v�/Q�`�g=��e�����Y���7���f�����=|��=�>�#\�b�V�g��>q��=Ͻ'��oü}?�>$b���"2>��	3m���>~�%>��?���>��>Ɏ���/Q=�ia>�!�>�&�<L7)����,��[����,��<�z��>�`�>�<��J���Ҿ��7�	���\�=�:����;��~>��ɓ>�/����A��[z�+��>k��e�<@O�=Թ���B��� >���σ{>��V>�:���⇾�%꾑~���g�>�">�!�>���*!�i㈾"�
?�;>~���VԀ�n4`>����hc�� �_AZ�tȽ��>���ʭ����>�T�=pw�=�A�������,�l��>f���i�,�0���K;>cBL<mu�<�k�>	3M�h	���) ���۾� M>��>+u��x��'��W��}��D{�4��E����>�U�	�`�y��=p�"��-
���<�v�>��
��ב���@=A鰻:X�>��?�x ��P>�F���>�l�ԳE�ӻ��E��W�<ܯ=m�>���>�N�ﲾ#�Y>�ϑ����<��>I��=�+뽳�=��?��H��`�˄?�XXp�ػ?>���>�h=D3=xD�=K����-�����</ھw,v�J�>Q�o�*�9>�4w=Ӭ���>a�>{� =�Fʽ���z�C�� �<=��D�P�LT�9�Q�>m�>q��;ga�=1��=����cф>��P�8�>�	H>�U�����=!��g=�>�/J�ч;�VF�����q�1>����q�>yv�(�<� �>�5�\ a�c&�=�4�����>�N>g'C>�9�>��>o��"�e�#d>G��>A.q���}>l&_<�"�����:;Y���=�t->�]��4��2���f8�V��>��>�2�28��Nb���ӛ=����y���A>a5V�p��=�EP=:�׽^��=9օ=|պ����=���=:#�>ط;�V�>񝘾ۿ�=R�J>8��>vq#�������"��d�?�?����=�z�P6�>p�=��G�q�2=�|�?�k��Eذ�4�#����>zSq>#�ݾ+��}���+�������>�0���4>���>�B�>Yj}�ض>"��jg=ύ>��:>)�˽�=��fM��}y�>�B��I�R�*'�:�쾽J��>^iY=[�Y<\=罚%���>�u>�<�ʭ$:W�=���}>�·�cEM��ý����Q�Qf#>�PQ>5}�=�]E�d�|<k3)�4��=@j�>��>�Z�>ᘥ���>�ڰ>,e9>_|Ѽ	�>�ȾcG���Й����qY�=�q��]��>n�>RI-?='=��zW���;��F>Jy�<��D>��Z>M2r��j�>`�Z�W��==��=���tԑ= �>�����>?����>U�>	8�Y�B��u >�1c>b>W5�h�>��X�[_�(���x)��TѾ�g�)\�>�ڄ>L7I�y6�숣=��]>�$�>"Tپ}\�>����ϽG>a�q��<��=b�V��%�=�U�w-ɿ�?�7�<%�>��*>���0>_$t>SC�K�X��XV?��Y>�X�>+��<H���,eY�]�}>o���!�:If�=�s��1?�P޾�5v���۾^���A>�<��Z�>�9���[#>�M\=z�>ۖ���=�	+��>�%'����>���=��s�>�%;�X�=R퓾�\�x̾|~辛rE�;�����T��m>����Ϡ�	��>�T$��=S�`�
�ۧ�=���>>�>k�W��ǻ�tί�_{�>&��>�?�>�+�>��=��>�0����7�M>�J�R^�.��>+L_���b����M�E>/�����<N(d>�?5$�>vކ�_�M��!����6?��DQ���=����Ƙ>CY�=���:<���E��>[%>i5$>dsռ~o�>]� A��1�>?���<ڨ��;=�=<��<��=�����#�Gb=\U�S"�>N�I�r=vO޽��%��w?��=~�?���=ݘ��F���¸�z�����>�롽��>û��
�>��==?�-ϳ�_D[<t�/��Wp�7O�=�RS>L�\=HmB>��%>7q�����f1�>B"���	�=bg��8)�C\>CƾML�ؚ���q;�)�W���w�>�q���Ss?�H�}�>뽴�9?5�L>�c־Ae\�aT@!���0S>�U���|�~I>ke��#J�����?
i��W?Z*o�n�"�>>�?	?��a��^�>�����Z�AH~?U�=��[��D~>b�q>M�>�]ҾM.Ƽm�ľ߉�>Pp������4���鷟�����\rL?�Ԭ>`�? X>:��<�M(>(ծ���>�۱>C�4��>ni�>̴?�5@��V�۹�;yܽ�7�=x�����<N1���(���D�j�2?f!���>�?�X�<Ɏ��>"F���)� �V??A�?M�e>��ľ.!?F�H>k}Q=��=J$�����>��G���?�ud�ܺ?�F���R>�����P��>�>�ko>��>u �==~/�,�ڽ�7���o�	3M�^|�4׹>9�?N(�LTO�Z��<Q��=�`}�-[���z�>�ؾ��?i �e�+��r���ּ����=A�¾���>t$��~�?�O�����^������'kT=��=J��-[+��ݷ���>��2��ܾ�Ct>S���[>R'��b��=�$>DQ >��^=�|?:齾��νH���_*��7���=�M��(D�>9�Ծ�3h>�!�>�.��[�	�7S�=���o���B[�>�*���Ӗ=��>�C��/��QR�7@��[��<��s��>Ș˾[�̼QlE��Я��*5�ep�>�S/?:;��G�k�û��'� ��Z����C>��\=Z�㼊�G>B�=_��.��>�a�"��oG>��_?&�l��4�>�砽�HN���>6׽/�=��!>2Z�>�#�=��j���v�f=T��<�V,�$(~���#>����xb�>�D黶����U�>bJ��s	��C�>}w+�,��>��>�O'>�H��&��>a�9>ižn�������C>�X����>/������1�)�yu���L����>�±>���<K�=��>K�ξ�о{� ���>�>H�Z?_��rP">�땽%̔�'k$?�(m��q�;峼���=�������>B>�>���>�6>���}�<���z��=�D���F�=�Ȁ���.�b/���� �=��>�d�=��پR���OJ?���>��R��=lͼ�<>�v�<��d>C�#����=�N?<��=ݵd>�����^��t=�Ϟ�m�/��{�^t��k�>G >�4/�=���� ���>|�==��=x���<�<@�Ē��ڑ���}�\U����e�r�`�[_���?�$)�Ԁ���j�i=5��m�˾$�>a��CV��a=� <��0V��ݐ�@0g>�"���Խ���=� ������U�e�!<��K=�:��">�0/=��0��ɼ��D�3�f�U��=�=�����>x.�@�ܝu���Ƽ�p<!�r�M��=T:��-���TH=\�=R,W���}>,�������2==�;k>ף�=4K>f�T<���>�Z�>�- =�� >�+<��c�>jh=�{,>�P\��1���;�=qɱ�p�!<>y��'b��1�>/��>w�>S���Ac�B�z>v6$=˹�>,������:�=�Q�?���rS>�%<��"A���ҽw����t>r52�io�>����;6B>�놽��=��!�a�%����>��=tq>�w<>��X=�.k>�� >�>`�0��ͭ>��ͽV~�=@�����%�����.��	m9�H>�O�>.��=Oo>��ʽW
����]>�Y6�rN̽;��>�(��ƻ�
�W>~��<���<"��=u ��KռA�~���?��;�%���&=u�3>E�使}ݽHG�l��>��>נ/���=>L��<�k�x�>|=��>keb>�v����=�5>�'>]�l�s��0�Ƚ���� Aq>�%��z�y�a�m�����=Qf#���N>�兼'���%��<��V��N�������=)��=��6�)?�>+5[>O���Z*/�I�F>c�ž��[�JV�ٽ<���>>�>> )>"�U>�
ҽr��nLϾ�B�<Н�ˢ�z�齕)�=+j0��D>�3�<\3=x�����>�=�H�=lx��f�;WCB�N�|>�Nf>�>Kȧ�qr_���>�s�>�4b=�=����<�D���=�?�= V:L�=���_A�=��>�����.��i��=Iz��Ӽ��<)\O>b�P��#���3<*;�=��>l�о�<徿��>����Q���0�t$> Aq>�'�=�������?�y��(�<�x���+��U=���f��Ͻ'>��>z��
p>�	�>��>%[ݼ�ߕ���꽪���,��>0�=s��=�ռ���$&Ƚ��M>��.>-���"��c�>i�����=��>��)���>����燽�D����=�&>ߦ�>�9�>Tt$@������>!�=\r�>Y��=58>ľ��?�!ڽ!ˢ��G~=���>�1ڼ=,T�����T&>�?���<�*�%X��@���2�cє>��;�ڏ�>����^e�="о�5�=-�s�!�<G=$��RB��ƽ�%�=?�=��n����ټ����o��=�j�k=L]h�b��>L�6>V�>Na���܏>Î��_V>���<S�
�V}.>6Y���>�*�rP�[ұ=r�"<�����ξ'�I�"l��`�>�C���w��f�=v�����<X��>�2[>d@��"l�>w��=� =�[��#J��8�A��~�>!�7��f�=���=�!?B]$���~<�<��w�>iČ=̖�=$��>��=ξ�����d�H��,���=�9����>������Խ�y������wq<�%�ˡ�>�\��5F�=�nO>P��>C�a2=��*�>��<d�n��mg=�a��)ː�h���2
>uuG��怾��T�h���S#>�z���c}��	?��D=5$��[_D�*:�od^�ϝ�z�C�ę���ҧ��h�Ŭw>}�
�o�=v���H>2=!�A�(>�S�=���<p�μT&�Ih���=���� T����_�o/�"���5Ⱦ��=��Z=nQ�>2���1�0�먪>����?��#>m��>_b�=zv�a����>�>�	��h�P��(9;衶<�:���=�:̗�nn��|���W�9���̄���5>�h��'�U>�L�=�?6�5�^�o�~�e�����ϕ��S�����"���2 {�X��=]3پ˻j<*�M�X�ƾ��>>��=��p��d�>�O������<�>��?=�����(�S�=��,>�m侐���w>PS>v��A
>�>@mԼ=�>*�=P�>U���ٽ�>����"=�ݽX��=�Xk>�c)��0��f�=�Yؾ�ᙾ'���~�ɽ�.�G���P�>�����#\�z��>��T:�=fNW�s�`����z�>�� ��ܥ>p#���O?=^0�<!�\?_��>����e>���1��>�l����>��z����>���>&�2>8���?�͜�dx��9�'��U���A��ۧ�=DL	��a�=��X=�A >�z>j��=����Z=:;>���<"�Y<>��>�kO��U<�Ml>\ݹg��=�x=����F"Y��D@�}C>!!
�-x�;�	��큆������=��%<�=�`Z�l=�<g�罝F���>�l���N>{/�<�̳����IҽI�R���Խ]3�>_����z���l���u=���שּׂ�E��>�_־��=�I�����=��j�b���"���,��>	켏��=��9�G��U��>���=A��GU��aU���m�:��=bJ�:s��A�>bg�@�Q��ĭ<�v�=#e��$�>��{>�b��d�J&'=yu�>y�ż%X��-�>;�ھ��[�(𮽊�\�y��������=�S�>=�>��s-�=�ؾ���>���=��zڼD$����䰾eS>-�;��,�QN4>�a�<�O��=,ԽT ����8��ۚ�*��֨g�*R�=uv2>���6��S>�>%�>/��n��>�aս��?���=˹�>�c��X�>�õ���z=�h@>Q�<��Z�1��ie=KuA��H�=����K <o�>�ӽ�>/�R> :�>9���~��T��2�<�:@���k=��L���`}�z���B>>aO۾�4ҾJ��
>�*s=��1�)=��>�3>ݘ^>B`���i���:;ٽ��r=����������Y����<�b>�(�8����ɍ�킁���нA*E�Pp1>�gP����ep�=c^ǼϠ�=����I�2�V��=�~�aT�=K�>�:#���>�5>�᳾�s	>g~�=�������>ɫ�>kн%"|={!�1�>�����=m�R탾D4:>�B>S����[>_����y��������>���>Lo?����'���hc>'��=�#�>��=�=�Ħ>!凾,}h�OPԻ�:D�������\>�V>��B����Y4�>��v�Ic�A+>ҩ����+>��˾���=C�>l�ۼ[�<�� >�ʛ>^*��c�X��d��I������=���g>��>[¾�K�>���=9(!������>{I�<l�2�;ߏ��0>�q�=w�>Έ�=��&�n��=�=]=�?��=�߾r���฽a��<�S�I��� )B>���s�X>��>6��>���<��ɼ�.t=d@�=T�?�U�>��n�W!E�8�>鷟>�_ӽo�߽��->}\?pwV>��">-[K������v����վ��V>�؈�m�{>��>��;���!@4�>���=�L�=QN��ػ�1E��6�0*�?�y�����Nb���N)���W=��/�ȱ���d=_{��OO���=ལ���������s��r̼˹�=��>�a����m9����ǽUjƾ�ǂ>�	�.�>!>�=?t�#J��ۣ�=v� =<�8�28
�x����3��p�=��Q�Ts��TT}��捾�9>{I#>�U�M���>Q�j�[%8>�,Q<��>b�">l�h���B>c�T?��L��Ц>�����t>�$�!Y�>�N�>bC���A>�����H�)�=�4��*oG�]��=���>��;*t>��%O�if>~R��s.�>f�C>E����`�!��ΪϾ�+�7��>�;��[��6�>�gd�h���[R�Eh�<�`�>�Nl��pi��N�>�_��DN������ɕ�#&>������>8���I0U��=��n��t�>�����#?'����ս�~G��;?%�=k�}�����A�>,e�>�Ry�lҾ��l=x��>Ve7�b�k>	G>�ڍ���V>�7����=�1�<�c�>�I��3>�É��O<;�*<H���RaL>�̬�i�=�j��]�־�+$? :���HE��p=Ӣ�=���f1�>L�˾�>)��=5)e>9�>�$�>�rH>��^�1�<�p	>m�>�د>��(<�'�=�1�>RI��l!�C絾�e.�~��lս ��>��>�g�>@�=4��Gl�,g��â���=�E�>SΗ=�$�>�].>x���^�>Z�d�Z�>k����d�Ҍ>�z=��Q�˹T>9[>m�)=����,�+�Ԃ>�> >��>�����9�}"/���a>�f/>3b�aT�>���;+���B'D=����ToM��A��E!��u>a�>�y=Q����
�29������ޓ>�+ٽ5�>�en>=���*;}��� �<����[����>e"�׼u����=���8_>XVZ>	mY>���I�>��>?�������>]P߽�V�>���>s��>��B?��
>� ���o���˾g��%�>�g
]����<��>��:>vO^>���>���>O#m�9�b=/ێ=��=����Z����>:�+>j�H�C�"O2���,�iŷ=/4?���s�����>k+��0ڽ}���i�X�A��>�N�>eP��|��;pξX-@�>�!>ݶ���2��bĻ:�7�N��\ �?�~��ގ��?t���g<u<ƾ�l
>͊�H�>M�N�d��:O\=l$��<k�=oGX>=�=���=�P>�앾d��=77�=c�>����e��Kq>cF8=��D��hx���!����؁���I���u�=�(�C+>�K1�8��=�ۈ=��H�C�<�x���)�v����=�
ҽ���<����ٍ�B� �3�ּގ��M�ݗ�=�4>��(>�߶�v�=�7���>�78>A����>�O>�4'����=�,����1B�>k}���)1�Ѳ��%u�f����=�>}��>_�2>Ҍ�>�ξ*t>>�����=�r��b��v��=j���c�Z���L�\�	Pӽ.�j
>��)�E&����em�=F��Ճ=��]�	�{��|�=�Z�K,�;%�:���w� �M��E���D�R>ϟ6=�����3Ž���=-�<����M2r=�]>�c����Z>K�>��I._>��	>gaw���=��?u��k��<�����>; ���>4>B`彼�O��y�=���=0K��C���ݘ>��ý���=)�->
h��#�ӽV���>aT¾.s��~��]��4�	>��A>y��"��=-x�=/�X=*t�=��<X�;�0�=X�-<w����&���F��Q��<�ǽf1���d����?<Ș�<%>�x�>=n��e����=�ᱼa��=7 ��C���}]���<����w�s>�r�=H��0<���>�)T>W	>Ɗ:>�i��Q��N>&p+��昽��=���D5Ž{�>9EG>j0�>��6���m���1=�4�>%0�;��>k;=C�>���Ď>�9��N�>�	����a�=��:�^�93ı=��bNP�Қ=��W>��8>��0�s� �4�>����bg*�f�d��K>����Vi>G�>��<K���fg�=��=>�G�>�[t=�%�={���
��>* �>��7�mŞ>�T�<{��;�p��"˾�e?��(�a��>U��;S��6ͻ<}�-�>y&;��J>��ӽ���>$E�>��O>}e>U0*�&�6�O���p�=Q���E�>�{>�
#�`<>�d=�����
���$>�q}��y��[�=*��A��J$?w>��p>E�?�P�>�_>c�ν��"��_"@�3(>a���͒`>>���=+�{�`׾��?�i���
��'�*�j�_�`��<�,>���>� ?o���Ǻ�>�m�t��>iȘ=/i,>D�Z?��������n����^?w�?>m	?J{����*=
��dO>S
?�x}>������ގ0�V��>�R�>E�?�&�0�!>�D��\�=�
�>�ױ�0�{��/X�L�>L���#g�5��=�t����Z��8=�X���g��R�>y�:���<���S>(F�=���>>�?#�þٔ[�	���h�>}�G>p>5��?����>7l�Ct�=�c1���ֽ��<ƿ߾M�>C��=�辌���!��q ������>P�����7=>y��*վ�3�vT�<`?G��>U�b�^׏��>�3>�i#��S>��?���<�����+=e���%?b=vq���;%>:��=�cо%�g&�=���<U0��v7/�v�|����>$��������Ǿ�E=�Ѿ�~����j�'?>?�=�-н��ξCs�=C�c�ƢI����>�0<>�s<斆�}$Ž�H�=41�G�����þ�j_>$���L7)>�TA=��ҽ#/+�X9��Bs�M$���>77��4�>���=�.�=$ւ>��H�
����>�{8<xE�^G���O��zǉ>x�H���>ڏ�C�)�@�9��Kz�0�?W��=�&E�y]�>[��>ԝ�=U�p?�QU>����6�>���㹛7�WӾ�<�>�(=	�?p����>�a?kԃ>G<���U>��>3ܰ���G��5�=�S=>�c�jj=�A�d�1�xE��DJ=�ũ��y'�,e�>��>���>K�;��Z���ʾ߉��Q=�u�>Á�=5]ϼ��
?��
>�?мW&�>?�n?�j���4�5^��J>���2�%�ڽ#־��e>�Y;>�d���6>D��ɲ�����(�>T:ؽ��>�1��\���zt�0g����?xb?�;>��>  ���]#��l�>�?��">��>a����.q�=
��=��G�˼�����=�� ���7���	���#?k�(~���?�e+>�
	>oI����;��4>I>�ɍ>�ُ�cb�>�?�l���t�? �l	y��񬾒��>`<#�Ih�>zF�G ?�F�2 [>z�0�;JFν�{;�J�>ep��q��_@���aO[>6W�<�(3>�3���P)������?��
�n�#=�;U��#?>���>=';�P��'�=rh���h��c>�彀� ��c=�;%>'k���$�=j��"��>��K<ō��BžCֽ���<��;��/>��=vO޽������?<n���!!J=��=m�u>+M
>^���,HS>�#>2Z��u�'����=nQ&>��Q�D����)ݽ���>1��>M������� �>w�t>]὾x���=�q�>Pp1=f���MG>yv>vT5�,��>9��>�9=�>!<�>ǈ;8/μ�3������d��Y݊>�2�F�ͽۧ��ۮ�ɰ�>�j>�=����>wܐ�o��=� �=�M�r�ν��1�3>TW�>$(����?�oW����/n��0>�v��~�=�@����E>\�i=FA0=82O��	��:���=�>� >4�~=�4�=�=N ����O
�\�=E��<�F��W�<G8��/�����+>�M%>do�2w��;��Aֳ���ް�>�у>EF�<����yXh�Ll>>�-�=�~>�D�>">$ED>i5>r�ν�۸�������F%>A�S#t<GU�>e73�����OA>�4��~��>G ��	m�>���*t�=lZ��e��>园�?:�=���=F�轹p�=1�@�2�x�.s:<��z�� ھ��.<?�>�>l]*��x1=�>�X�=N_=&0�Lq5>�(>���W>�>I�ڽr��<��G�	J>��>�{l�}��bgJ���-=>[�=/n�jj���=aTr>y]���>��"O�,��!.���� �i>_FQ�:;�/�����=�
=����ѽ��$�>�>�9㽲����X1>�:>g'þ��,���>��>��f>�*޼���>8����\７ǚ<#��;ƢI��ﵼod�>O�.>�ݼ���"�^�U�¾�>h��>VHY���H�==�nt��G�=c�>�e+��Ÿ���~=�M�fOB=v��>: �=�>&S��}ռ��2>�q�>�K?Ll>>ŏ1="�۽����Q���A>���>��i�uX��v?V�e��>f�
/�=>�p�
�;3�w�V��=�q���=��n<Z��k�P��'L��/�>2w��_�R��Ӿ�r(>Ӈ�>t)�>SM<�z1��8�~t�>GZ��0z�h�?}"'?��@k�(>&�>�{�=�#w��u����=�������?�������y��蚽G����t���>@�žA��=$��<�ʄ�M-�>&��=�u9?� >�
ƽLTO>��[��J�����9<�y��%�����>���� >���<&���)��y�����þ�rѽ��?*�<���u�����>7�>M���K�G��r>fk�=�6�>xK>��0��ɺ>��ݾ�b�£m>�
?�~{��$�>������3=5�>2U?��>�R>GZJ��=>2w�>��`=��>W�_=$���Ș[�z�ü3c�r���#�Ƚ�>����8��@�����[%�����0�w��>�����?�����S�>'�C?4���?_����0�������?R�> �/>L�D>��[=Y���뼂��>�k>��u����k� �>�>>���>bJ$>�`w�F%u�3�Q�l�|9I��>Á�>[?���&���O)��9Z�(��=J$�>�-�=��从��:E��<h<��,�=�����z��H���rs����>��޾V.>Uj����ξ7�'>��7����'�˾C�=H��5���\����>a��>�=���<_�D�5>_$>�R�'N�=i�<�㞾"TI>Xp�<Gr�>ę?��r>R,7�ߦ�>m�*���>����y�����?�0� �>I���z���('���!�ً>B[.�����z6>����Mjh=�=�'�Q��=�U<���->���=�o���'��|>�u>�*��U=kQ>�Z�>J=.�6=�E����j�n�G�����d@V�L�*>�[i��U=��*�!�����>�>Lݕ=�/�>�pC>��;o�6���=�u��
Q>�>a�;?	��>��j��B�=���>wJ'>w�������!T�c�>�!>J�>z6�>-!�>p��=qr_>����w��4�	������=�T[�I�W���;��Y�>�<��H2�<���>�J>���_b�=�E����V>��=1|Ľ	m9>��>��������^�>|�>7��9_���9>t���{I>�Y3��2��.��=������ξ�_?���>���=��Խ�ۗ�k�>_[>�g�>�z=]�H��B��Q1.����>��=$�ӻƿ�0�{>9����Ty�G=D�ga/>�
¾�`)=h쫽yX>���>��$�X�G�]N�=��>X�
@�n~������ ?�s��(���R=�tͽ_^X?W�D<`�>;���8����#=����	=����<�����>�0>��<ڭ���>�2=ˑ�p�s?f�L���*=1��<��N�D�̾��2>�L>�ʤ�Z�G>5���S�=�(�>�D��·�$����ȶ�[?�����5>��*�Y>!����߱=�[�k��K卾τ�����=r��>z�?5$���=�q�>�@b�#��>��?1%��G�>��>�rpɽ�a�=k-L<��y���?>c��<���<Z�= tX>�L��w=��9�>��>�� >Ve?���νS@��Ac�es��S>x��=M���_��W��=2U�>@A�X�F=tA]>�������[�1���,>p���J1��e��</n#>,+��1�9������=����>���>U0*>q��<1���Lp�=�FC��I��� �ı.>>�=7�a����W�ɽt�@��D�=w�>�M�;4��=���m�������q�>[Ӽ��V8�s��<�`�>�w����q�'�Q���1�>��v>i�b>���ܸ�=��b<�]�6�/�c<f�����=ݲ��1%�=�#�<>�0>�o?��Oͽ�ס��s/>�A>H��>׆�>�����E<������G钽4K?��Ǻ5)վ��M?[^�<ޓ�>�x>3P����p뮾�р�QN4�aO�xҼm�>��;�|��=@���>�G~�k�L��:>�d
?�+Ž���>���=4����ع��+b���c=A�׽f�:<5F>B"����2>��;��� ��]½:���ٽ$8>׆J���Mܪ�$���۪=����4j;�8�=Y�?.V�.�Wx�=UM�=�җ>4�+?Vνvm�<���,eٽ�3�>��~�����8������>ڬ>���=�����>x(j>�3��f:�ę��������>&�2�V-�|D̾�x�<��"�f=���
j�=�$���^)�Kv>{�G>�����>�x�>� ��~=aq��̴����=�>��ψ��0�T~��������r�;2�>���>9�>��[�l�;�1�?>
�����xf>AH�v5�<�^<�0/���-�=S<DLi��&����>��>�$Z��(=u[��R�=�8�><�=�u)?��<=5}6=�7�=v�*>����Di@�tV>�y�>H��=)����/���9������-@7���<n�۽fk=>N����h=�������>��?d�7�A����R��h����>���G��>?E���{p=�-W=�}��>�lx�=�D��@����=�E>�z>�5���=Ӥ��0f�=��0�U��� �`;�+�>ض�V��=pB�=7�>�Y���8M=�Q�t)�>��3>G8�����F�/�S��X���%^��Cq�ŏ��ܝ>��B=�O���s�<��>�\wӾܽ��V�D�;ӼC>����؞���F�>\rt�-�3?��g>R'𾗋>�ű>y]?<[�����e�>�|���6żi5�>L��W`H��>ʉ�>#F>����ܠ�<䃶��{�>x(�>a����O>>v���μڏT����죓>�\��P#;Z�>��>�r�>��*�NԲ=�O:�ŏ�>�6���>(�1�=���=������	����<�H���>�߆?�>����¨�R
���4�����>fٓ�v7>��>�����f�>�!��n�'��3N>��bJ�>D�U>�-??TZ=���
,>�W=g.�j�e�?��0��eT>��>�?HP�oX=Q�[>�U+>����U�>����I&>�y�>��Ӿ\��>u�m��&�>�/=�7��'�'���=���Zأ��Kc=i�>+�޾W��>��x��{?�J��	mY���i��"a>��;X��>��p�.Vľ��/>/ޏ<�򾾟B?q�IK�>#��}��>���Uk=b-��To��O���"=B^=��Ľ�7U=����|
���lǾ�DX=���>b8�Y�ﾃ4�>����L�>�昽�]?7�?���>x���ʾ��p|���c=AF �=
?�o�
>�PG=���#���`��ضH>r52��쑽N��,-�<˾K��n��*?�+���tZ=��>D�=�0h=�Z?�'��f��,��>���>���;�E��5c>і��]%>��m>t)���F��֩r�l[T>���>�@�����@+=�GĽf�?X����\?�F�>`�>Ynپ�]>S��>�͟�7���&�zS�����(��>��bG�;+M��?���"�Y�������3�>^�O��嬺��7����s�Ҭ>��޽I�=l�V=���=[�/@�O�>̗�>j�;82O�׊6����=}���?��::�1�?5>��,4�ɓ��>������;��=�/��7T>oC����>ܷ=�c@����U�n�z����2Z��Ė�d˒�߉>]����=�>�Z<�E�8؛=jO�=�x�=Z+�9*�$=񺎾��I�e�i;��T>NzϾ�z�𧖾�y>,g>�4�=�Ͼ���>a��<dY>��&>��~<zǙ>b�E���$>ߦ>P�L>U�">�p4�3m>x$�:��<��нR�=΋��}>J��㷽6\��0Lf>P¼�
K�!�l=k�	>��@�Hኾ�(z= ��=`�>���>�{�IK�>�U߽ʦ���)Q�?�C��~�<)�*���#� �>����[�Z�����:�1>�2�;����Zm<y]�<��u���;�f��>j�>�I�>=�>���=��>n���!�;�3?>� _�Gr��a�=n�3<sK���0�=���5�L�1]H=��>a>������J>�m��?N�+�ξ!�R>�=�:�ɰ��K��(+����%���s��2;�=�)�>��1>j��0��n��U>�Z��ظ��eW�S�>D���b���ѧ=7������"���ui�8S�m��V�=77f=��<+O�=�r�p�c�N�>\�j^>�3���=u�G�	my�_�=$�B�L�$>U�}>%/��O=+>��>h�$��˼�eN>8)��Q^�Uٷ���	�D�A�\UV>Q�)�Zd���b=�`�=���>�}9��U�<U��<cW��;���D��m��=3m����Ҿ��A��3>�!?�y�5�;n/=#�l<�ׁ��P��ȼ"]>~�<Hߤ=�=N��>!�a>hа���־Ӽ*�?��=i���!>Y0q��L!>7TL>�Շ=��>����>�I�>�q�Y����C9Q<=>Zd{>y>���v��d]�� �|�W胼H��>1%2>��<�=3>�b�>�E?>e�@�b=`��x)>`Z�=v:����>*�c>��.�lj�B�6�x��;B[�=n4@>�->�S��7 ?Rb�= <>0c�S]@��|0>��->B>��+���>(�z������B=����%�<�zN��&M=XX0=�b��֨g>��=�\��M22>Yݚ�-왾#J[>�ʂ=���=�Xw��ý8;@����S��=<%>��<�2�����d$�*W@?6=h�uŽ�����& ���9>Ll~�_B���=�M��d
�^�>|,]��x���=@12�!�8��9�������<h?���d��T�_�ȷ��,�>��=��=�Ƨ>�S�cA��?>�	6��0ż���B!>Vס��e�a=����h��֍��q��
�C=&�7�%]��H�>!<�>2 [�w��)yU��^7<�$>	�w>L�>]ħ>�>�b�;'h�<u<�=9��=(C��})��K�>�A�;TG���<c�=֭�=�j��N��>���ʇ>�c�=<E��=�` ����>�Ĭ>Rv��#J>����zp�������0�d������r(�d�c>�A��io��u�=*RA���ӽ�)p���]�c�=�GD�}�ͽ�!;��e=����2w�ǝr��x)=�=иP>�#?��X,�B>>"o9�򵗾,}(>9b�����<��|� ���\
>�R>�{p<)"C>$���l&�=����G�a�>�]?3�
�߼�(ܽ�i��<f >?Ȳ=2̩=bء��5��s�>Z�C<�
�8����OP�`v��]��Z���b�X�A><K=��������9h�X�>��R=_҈>�;�Ι��L��l�˽�U?�[�����=���#2L�����/���p<�ax��J���?��^4�� 6�,�[�'���N�>|�x=p��>/>�kL��O
>}��=��/�e����*���>;�u>������(=��a���F�=^T>r�)�$�����=7OU>E�����	������\=�>y[i<�G�=�$���&���<�̓>��=Aը���|=,�`>�e>D����>a)�:�>>�D��w�߽��>�r�=�'>��"���<� ȽI��<_�>6�>D݇<-���5)�>6<>k`k��Q�� �fZ���!>V��>�>��������ګO�^��=�7>���b�v��B��W`h>	P3>fL�<��>��Y>�Z>/i��9(�=P6?��>Y�/>�yJ���<�ܿ��Yo��2G>��;�g�;�$>J?N��>�"�<\�=�k���{̽u"��ܿ=3�1<�,�>�����о(:ܺ�j�<3�=���=���<�[����3�7>�9�=2q+����>uY����X��>����`\���� ��;N�1@�P;�2%=��=�]��7=[���HH>��?��G�5
�=	�>�f{>����#g���>`�g>#����<�ˍ=%#�����>����=|,%?������>d���O����L>(I�>\�>[�K>W�>�y�$b�=	�����+>TR����>�>���=�Hǽ�7��sK�>��ǽpž��%=%���_�>�B�|�>+ǾK=#�?_^�>kP�8.�=wJ'>R퓾��|=�����s�<��=�v�D�2���D��>ᗊ�o/����p>Z�y>��=�e��v��ɽV�T�F�&�2?.9=�_[���Fa����x��P>��ĻF쓼���"�p��#�>�*3=���=�E��B��>*��s�n�z�?E�>]ė�C������m�>�ϼ���>�i8�&s��W?`YI>"?��>�1�>c�:>��?�Xq���ǽ���>s�>~6����h�d=c%f��_�<�T�=mʅ�� �*�����
�Ӈ��e=�H�h�������s!�9�c�-�=���=ʦ���N>�8�>�ɾh��=_$��Bxt=
ד����=$(�>ܝ��O>��c>O�X��V�To>M�_�nޘ�g��@�p�S<��ݽ,��>���wh�<k������U��x�Ǿm�޵>�)��?�XW?�!=���Z>3�(�CV/��-�>Ḍ�t ��ԛ��a>�����	8�>��>NEJ�s��n?J�>E�>>��!>D�'>��,�:;����C;
/����>�z>���4�=�L6�����:�>a�>�?�=�=F��>���酾 �R��̾T5>��w>�z.?�Ž�����'�r�=S��;�T��ϛ>/Q�I׼��3�����X�{�`귾�ܯ���S��s�=�,#>��Ϊ?�c�>5�ǽ��	�,�+>�����J����>��Y><���C�=Lþ�_->p������	�>Ƨ@�7�h��l?�Y?�k��\>!�>=���=L���O�?�=�ٻpB!�* &>�i!>*�y�Pż�W�<�jjq?����!��^>�w_>8I?��8�A�Q>�ug=}�>Hm>���=��v�v��wM=J��>C�c�K���E�<��=��<<�>L��N�>������ё\�W�)>�:j��):�A,��	�@@�2<M���?�>��ľ^�:�<E���D>�˾�/�����>�!=O��=�7�z���+U=�P�od~�O]9>��?tA�,�>��=�ŋ<�P��Qս*䪽we�򘡾ỳ�P��yX�>�3�;�`.��2�>K�3������F7=%X�>�2�;@0��c���q>��=#-�>�=��,c���X�+�=n�¼�y�1�y���=�?�:N=�؜��y�<D��=��?`�>��>˄�>j�=�i,��g>
h>�&���\�ʌ���S>��V='�[��T��&pK��8=�]��z>9c�Q0�=�U>�50��B�����>k+.?F�ܽ�ܵ>^&�tBh=o�p>��>���>���=X�F�Z�=����=!��ij��Rl�N��=�g%=�3�=-	P��u}>��ž->�>�C���L�<��=�W�>�;H�tA]>q��2V���>�h���־g)Y= {�=�<>��>�5�� �=�I�>�׎�-��=77>dv���5������yѻ��9>'�[�LO8>�8=���0�=����� �Cu3=Ͱ=$��=��=�'���
��J+�UM0��}>()p=6Yþu�<%]������ᴠ�I���)��e��:�7>��.�'����������������U<���&ǝ=Ii6=x�������꽤��Zػnۗ=__뼜���1��=>��=�vD>�ڔ>��=����yǼ��=��*>5*�=1�����9>=D�>��O��}c��P�|�"O�>4��=K>H�Z���>���>��>x��=0&"��b�uˮ=��E=�2A�X�U�������+< ��>_��=�c �0�>��ƽ��>O,=(�:�α��w=vTu��7�=o�=J">��R�"�6��T�>C����*f=G8->A��Y0����]�>i	>ٞ�8���8����"��>�A>�}��¾�5M�KJ=q���	ǽ9�E<I��=�'>��>��M>����*�=�K>U�>��=�&">��
>�a�>^c?�eH>XV�>�?�
�����>B	S�����\���j�>�0�<��=0>�H>z�=�5~��+>��e>�s,>wI�=��o��HH>M$>_N>���>I������G�E>"�>�I>c�>{�u>�o"�G^=뭑>�W�!m�h=|��e> �#@���<�;�=l&�>�B4��e>pA���Y=�k?�
#�_)+�A�>�����aX>��=I4>�Y>ު�=�ה�C煾:@>笽Xr=���>\�t>B
=�>Aj>�?���=:�����=�4�>aO�F%U���S�A,�=�e���=;S�>'��������<��h=�C.>��W�^."�cy>��0Y��`;�O0=l[�=A}K�b�?V��={�<�8>������P>���>���>�Kj�y̠�V��<�De=������ɫӾ{�=��>5^>iRj>�,o�E:r;.�=�豾�-��  ���ɕ���@>�9 ���>���s���z�B�7����>����Y�>1�>�+>ᗺ����=�R���ͭ�M��`v�=.s��y\��?$��=�ܕ��z�=^L�<�����lvd�q�>YQ�����>[|*>�c#�/4�=�Q�<��=���=�x����#�)"C��%<w�Խ��=<�>��9>�z>? 	��D�>��?d]�>��k��ƿ�>����=}>=�@�=��j<����<e�;�����&��=�DS�E�^>�nX<�����N����=�/��h굽��c�����OZ��q��4K��?5���(>9�">���>s>���>�lJ�;|��j���2ھ�3�-��<O柽���e���L{>��Q�(���߾Y�=!<�p�N�`��>����.���_@/�ƥ�=�Lu�X�>l&�>a�->��nL/����vƷ��>=�j~;���<��=]� ?0�����;��G�����*W�>.s:��$";1�ɽ�b%�A�����>���<����x�<�F �?Rľ�7�=��?>�g�MgG��j�=�җ>�)�cE>�_�>o�>���=�Wm��sƾYL��%#��m;��� ��L=���=*�C�s=1_�>8-�=�ھ,�U=-x1>�W��h��:��:������(�m�r��=gD�<'��>?5�=-y<��C�=��?ޏ�<F3���;>���>2掾n��>+h�<{�!>�X��ڬ?E���2d�������a��:=�s��.�,>�|?jG>���>���>Z���z6�����	G�^������F�=�߽�J>X��\,��+>�_�	pz���F�?�޺���>�_0�V�:>����^e��WC�>ͯF��O�>.s>�8>�@h迾���>$(����d����f���	<��?����y>���>�ٽ*����Y��@/��m�I>������н��v>���=|��3���w��=���>�s�>+������\�>��>%��>Y�8>��
?�������~�#�&6�=�/�H��>t%���+ּO� �%>7�X<u<�<�
��=N��;a2�!�|��b辖��>�`���
��k�����P>'�>�v��pƾ1B�>i㈾�N��cbs>.9.>=7=�z��~5g>�a��n�>�Cq���I>�x�>5�Ҿ���>%ϕ������>|'��[оl�l>���> W
?W?�̴����N>��">�1z��y>bX�/��0��>?W�'��f��=e��u���!?��ξ��J>���>�<�>{N��[>_�^��K>6<���l>��d=:�4��>Ts�=�d>��,>��>d�7����;���>�� �o� W�r�)>b�>L��={N���X�jj�Υ���=>��=`�>��\:�����C��'ĽY]>!���#¾>>a��靽йۼ�\G>մ�==��P�	�w�T������)8T�*��>	*>"`>rP�!Y�>�=�p`�Y4]���<�S�ܾ1�J>9q<C�>���t��<k�	>b��>]��<+j��>R_�]�L��>w-�>
��C�ڏ����=ׅ_�r�@��ܖ=�y=�н6t��e¯�/��=y#3��n��K��=�$�>>6��:,�#��<}�=�;m���"4��1�=�ۘ>vT>ޮW���>��=���>	��Zm���<ƅ����]>y�?�ƾT����K��:V=��ٽ˹�=�̉��@e�����Fyf��8=.� >��:y�@>u�P�=��=��L��.�����8�+><f �Z	��/>�V,>��<rĊ�6>���>�A���3�s�(=s����#�z��Y4]>X�Ҿ����>y�=�Ԗ>���>�B�<�A����=�f���>���\w�>�7�{ʾ�ҳ=��Z>6�?ܝ����>4.�'¶�%�=�<�	�/�>�[�=Z֝����>*Ů=t)�>�J��Ġ�l{;<��mV=>����+Q6���ҽM�M���ӾY��>JF�����^������g�����z�>��:>�k��汦�zߨ><f�ʤ��3k>��Y>+Mj>��?�E=��8>DQ �M-[��T��=�߄���?�
���ǻ,�5�ƾ�r�>j�k�iᒽT����`>76;=5{ ?�t>=I�=S"�d郾R�>�nW=�ɗ�+��e��>�pʾ�N�=x.>q d��8�>�	��Q7=�i�>NEj>�/�>x��>޳=����kn��κ>y;��m�ս�N�>`����>7qҾA+����t>^'���>;S�>
��<wgm<�> �����>ŏq>4�ý��~�Q�>Ӥ��W��>�2����I>m>؛�=�1?l'�;+o"<Q�	>��ؽ�۾���`�!��>(C?t$�xzžb��=��W=�F/�U���/P=�A��"�V>�Ӆ=�:1=5H��ݝ���R��m7�PQu�a�)��؊<�#>�:�>]mžC���x��wJ=��>ٛ>����i��,?���>J{c��\>"[�c{��=�᾿(=������;�Ͼ3ͽ�=�>F�>&S%��S�>��?�j�>a2u>\Uv�~5g��-���s����+>r����4>���==ZA=���yƾy�=7��<�*�>V�=!%�h�Y>I��Z�����/�$���N>!�>���{a=Q���A+P�du�>�>���[ >v	�+=�#��b�⽶�T�]P�)<�=����I.>�gl=ގ0>�~
>%���ö�>�<�f9�{1���H��Vk�A��=�n�>!̾��ս{k�>��>�m���8�>L�9> �D>u_>}Ν=Ho�=J^��A�����>e�Z>�D�>g���Ӿ��=*�� ��=���>�����#<>���>2 ��|޾Gr�/��>�+��C��"U�<m�~>e�o=M�����`�'�����> :�>	�>�#�>��<���>r��=R�=K��;���]�<.t�=�>-=���=X��;ѐ����?=X�=�"��E}>蟐���=��<$��:y��a2�=h�f�h"�>+^>s�7�����0��$E>��m>(
>!X=�s�>�wH��N >)�a>w�>eV�$��<���v�=�3ؾ;:�<��?֨G�)�<��x�>nLO>�?���>����@����a<��/�0d��@���A>�������������>��ɾ�3��P�[��5F+�΍)>�g���>Þ�>��]=3��>�� >%�����f��>�MY>U��?	������;}��mʕ>EG2>���>�r�>,e�?���[��>?>��%� <!\��?�8=�h����>��y���>%@m���?[��>{�f���=v�̾��r>�J>8H�=nL���>f�;ga/�b�>~�������G�>v��<�d��lC�>��?ܺ��9��eL�u=�G��(�>�>�E>��轪H��a�޼l�>(�����<p �մ�=ԩ=L�>0G/>���>&0���=r2�=����6�>��a>0>~�K>��4>\���> $+>P�2>f�$�Ǻ�>��=��=�V �W&�>8��=�{O�)�۽�Gj�p랾�Z�=���>���;�L?������>�b%��շ�#�;�v��ٿ�ʉ=����'��� �=���>������V�t>RDF>�̽N�+��ɝ��x�>�X�ҩ+�6<��XB�A��Z*�=�t?��>D�����¾)^�Gc=}?�=K>
�G<~R�>mS�<�:�>���?t��*Ⱦ�4/� �=|�=�S�Ӿ�X�Qk����>�2�@8>�-s���]9�;�����>8�a���'�F?:z\��%>����8>������)�v�S��>o*>iɣ���n��[X=N��>3�Ѿ�ۗ�ޯ�~�=y\>����U�= :?���/n#��Z�����T5�>��3=X����}����=�-�+M�>[_?+�=��a>U�q=9�9>3m���g=sh����þc�n>d<�=Օ/�4����`��W�>y鶾��K:������7T�>�<+����m��IپS"	?�>=��B�\�,?u��=m�5�|׽�+�=p|���z�
�>����,+>�;	��>��A=l�>\��=Yir>��� q=��K=<��=�h�=5A��[�>Ǜ<��ݾaqؾ��>�ԩ�������=��7�ŏq>!����1=�:�=P�]��7�<?W�=:]>#-��[B~>���*�����=�V�=��� i�;a�'>@���Gw�>���<�8�>���> C���>�#��O;|�Y�޽E��<N1��&<k+6�vC>Sc��:�>|a��u�?��>V�	��4Ҿ�L>��3�w-���R9�6<�>%�Z>5�F��'���[�"O���8e>�P�8����y$���^>�Zs>y�=x*�=Cs}���=������<b�.@�ֽ=�Ѿػ�>����襾�ƾ��>1��?qU�����<�~I=6�=�۴��x��?>p	@����>�fq=��>m��c(�:X��<N�=Q�=׼��>9���zh����]᭾T:���H̽�OD�O>�R��෼�F̼�-S>��?��/<�xj<y;"��P?��>�Z�<�8\>�/���B��fZ�����V�=�>8�B��0s>8==���>b���<�'���rh>t;��� =������=A� ?��>q�7�ʦ=�*A>{I;?�I߾����T=�t�/Q}>��/=��p���>}�W=g�q<�-���1=^ľG �>a2����)>Nbн�>+�V>�'Ͻhy�и��+�N>Uٗ=��ֿ �->�Aپ�/>� �Ҍ�w1ͻ�r��m��1�����=�=�X;����=��]���=d@�>���%>㥛��;���k#>@��>�.>I�V�L�>�nȾ�l�='�þ�O$=�3�>Yn)���"��Wm>���=߂��D�U��<i9��1��Jt=��b�E�־_����A<���>���8�*��}�����=*t>g~�>ѭ������U0�>M�N�����SC��T���J�
i�=`�_=�V=q��h�_>"��>�k�<2=��ծ	=#k���<R,�>�?M��=�Y>�'����l��'+=�i~�Lm�<ף�82>��>��;���=:��Lɽ�� ��`�>l�>�~{>ȗ�<v�b�+��>��\�Dlp=������=S�ֽ5?>e�>m�i�RDF�5ѧ=i5佉�x>y�T>��+�ǝ2�|�>�!<VF��'����<>ʖ>aR|�!e����>�=�Wm>����0؍=��e�L>y]߾g'C>��<���qǛ�4�=�e;Q�>��>E�>C�
���=���>�W���:�c��>��=1Ο=���UM ���	?K<@>�� >�H�=�7{>�G>O��>TF�|�>t��bJ$�{14>+�����>/��<���=H����&�q��>o$�>�>���x>Y����F��7�m�p\F�!ȡ>��>�����楽c����>�+E�uV����l쒾3�E��ى��Y	�5cq��ք�a8W=����{����o��=mm�;'����;���=��<�\8P>l	�a�����Ic��eȱ;`Y9�0ע���@�[��.�=���>]ć�G9�=��;�G�<#??R>����\�6׼H3��w۵��W�>nھ���? ��Ou0?�y��^c羰搾Ps������@j�=XV:>��v#�:��>�� �7��q�K�]������=�u���.��N�>�eq�O�o��u��,;=PǓ>;��L`���>��#��D����hB�:Pp!�G�>0�3�֭>�s�=v��F���������+�=T:x�:�=�F�B�c�����>9<����vTu�_վ����T����4�>�ʾTH�г�=�Vþ��?�S�����4�<�1!�<�־*W��V̽�Q��ʽ��=>���yھD�U>��u�9{�=jOI��=x^�<^K�� �I>p^�=�g0>�����_������IT�|ξh����a�>��0���~~��W��\�(>-xq>�|�>�4�P߂>'�5>|{W�L����>Q�l���>���;a����t��9E�>���$���8��Pǣ�)?�)�忽�w������G��"��Ǣ��S���>�>�]���l�*o���<�=�T>�3�����>F$�=y����ɽW:D;T��9��>'K-=�$�<y�>��=2 [�lт=%X�=����'1ؾPߒ�Ɏ=�E>��+����>#I�<�_�����>*6f��?ZG���5�>���>0(?	'>s��>8���S�1=ղ��u����>&Ʋ<^h��ۅ־��=J^5?uY���*�>˃�%�꾫Ϳ=b�(��2d����7��>Y��="����8���|��� ��5q=K�m>:#���V��^���������>�|��	W;s.e�P�C=PS˼-쩾m�	�<L��xG�=*����(0>w�ľC���(�ӠȽ|�>d��$�e>�
<ҫ�=�WV�M�Ⱦ)��	-��u�>u_>~ ?Z�2=�7[>ڂ>S����;�AH�=�ؕ��I&>&|��W[q=ı�0��0/@=QfýN��;��=�����^>�D�>�ՙ>Zd�EdX�N���RH?w+=���>���>X�$�N��=#�W�i�3�>�ߛ��|�>k}��TW~>ԗ>[���4>Ed8���=��>-��>o��>c�.>L��=^�]���&p��82O��?j��Ίh��,{�Sy����>���=�>X��>ގ���>H�Խ�ͽ�[�=��U>z�$@̴-���=��>�W�:<>r܉>b-�>Ь?���~�k�b���9�Y�=�
&�w�>r᰾g�=�,�>/$<���U0�\T�<
���t^C>q�>l!���&>��=����#�]ߧ=Nb�>q�1�����-_=x�>G>�ô����=��־��>�VĽ�7�ǝ2�Gɋ>�OG���>���0�'>~��	4X<jjٽ�˸���<�	>�_�<����V~���%�=���>bi�����>k�>f��:;>J���:A���'�|,�ܝ�>Ic4��`�>�����C�g`���˽ 㙾L��>�@\��͗=��O���>� þ��мg��>8��=h˹>m�o�Z-��֋�WA=E�>��� ���/?6����_Y>���>���>b-���%>�i۾r�=j���|^>Է��Ͻ�>���>���$������i���=�3�R,�>ӽ��f��=���;i��><f ����>�to��;��\��0L?�v>��U�5)�yM����>�m���'���v�>�����0
=�K>k���mq��v7'?��V>hZ�=�i�=�r-;'����i���R=�o����>$�<��M����A�>2V�=��=�Ⱦ�Ƕ������>>��>Ԃ׽�>�=d:�<���r�&>�|�=YL�>T�%��~{���O>;S车*m={�G�l>N��:j��g���>�_ͽ�T��A> �5�������=�.1>m�=>�=�ב>~㛾M�y=b��*���^6��D�t��B`�>h�>
ד���>A}K�.�@�fK��J�1��>ms�=�,�vq�:#*��)�wx<�F#�-[��K�a>�@+�/�t)�и0���=��<�Y;>�ʨ>V9#�p�����?*6�;�ͬ�)�6���b��
F��>3=�g��	8d��T8>ܓ=+�>�Hg=�>wr�ׅ�_�=(D`�]m����L�R'��;���bg�i7�=�:�>��>���>D�˼"����>�I5>�Q9�y@�>���?o*����)Q�d̝�6�{�ѯ=���#����9���>��/5�褗>�@�>��;ö�>'��>�a�=ձ�4ؔ�(bQ�RՔ���[=�t�<*,>!����ǽbf_���D>Uj����O�TF>����ɫ>V-�d��>�V>���Ae���=ػ'?��<��=��iU�=�@����W&<>A���Z��Ƥ���6���?�5s����=�#>G8�����=���(�r�k+�>C��=���yX?U0�ɾs���B��5=���>V������+>���[Υ��/�������,>�g��%��=���=�h�tA�M�%=�L����j��Ƚ�ܽ�Ň�פ�=���=D��!巾��<�]>�<>n�M�_Fq����>�vd>`ꇾE�?�X9>[	���q>�I�=���>t>�ʘ=z�>s��>|�<����Q��m�>����F��>-�>D�,>��=HÉ�}AK���?�>�J>�'�=���=�G{�����d=#.�=���<}���3>)�*��)����=y\�V���>����<���p�k�w>k+>ܺ�>�f���y>��<f�>�Q,>q̲=���<�7>K�ؾ�B�>��h��*�\V��?��>	4�=?U��~�Y�)��!<ڽ/lͼBO�;�^��KZ����>�[L�4�X��@��h�����ͿS�m>.���ڪ�GZ�������)�Q�����>����<�4���b�(�E؀�M�Žb�>�%�>L��=��W=���� ���,���C��P+=tA�>�4Ҿ���k�þ�9?U=���P�<
�����a����wL]<�X�1BX>/��=A+�>/Ī=>"�"7þ����p��J>�s)�Edx�ȗ�"TI���k>���=a�~>;S�}�=�M�=xz���r=qZ�=M���AO�9����(��&�������=�*�ǜǽ�Wt�^���y��۾��:�����ͱ<=���JP��u�U�|��Ͼ�-*>�OM���<���>���>!Y��,�:��$�>G8>fk�(IW���>H����g>��>�4/�-⾵O��m:��S��>`�;>�n�=��a�\�T>����~?����>�ۯ<h?�>��6��,��s��>���={k ��A�>�]a=W`H�(k>��RD?]�&?zp>���>�#���=;��;>�X�>� �O��>;�:��ZƾF|'���M���>�n������(~l>�ɐ>*R�>��2�v2x>���>��R�k)`==����ʃ=*���D�ž��P�m�U>�M��'f�='�=��I�5���1���A>���L�>f����>N(�>
�z�O����G�����_"@>����*�>9>� �P���V�ۻ&��vqS?�2��`r�=���>0�<�o�0��>J�Ͻ�	[<7@>�����~�>~����=���>h����Z�u�<`W��`Ӽ��<�L��oӿ����CS6�7��?��+k>�b�>��/�Ͼ2�R1>�)T>����T5A>zR�=�a�<�>p��>W	� 7��)�k�(�5c��Cr��.y�u�G>�>1����=��T���c�>n�x�1]�<�A`>�=�6>|D�>�?�����$ ����>Z��:ѯ��̾@QY��>�z���m�>�]F��?􆻽[B��g���y;"��G��G�A=�>0�A>����命,>0�w�VC=�ȅ�;S(��{>#�Ⱦ8ۜ>��">P�R��c�=˹�>*�Ӿ�X?�ܶ=�-
>��(ܽz�����>�'$=/��>�վ#_.;�*A>�>����ߜ;,�c
V���|=����>\�fNǾ���<��>)���^����>肊��z��0��9c>�*;�R=\�9b�/= #����>��8���U�a2�=��>s�`>��8=<�$���ȑ�k���;�=帳�o�����=�>{�����=�4`=�����=�c��Q>�����ʾ'k�>�Q�ٙ;�4,>i����J�g�����<c��=���TtD>�U<�#�>�T�Q���������~>9H>��>f��xE��~i�g�M<*:�>D��=����>O>msC��2�>�X������<w�>��?݁=��X=H>��Ľ�/�Fr�ߦ�^��=��>��L�s߽���e�O>�����Y�>t�a��b�>r�)>Y>C������=/n#�e��2��-���?�=���=ЛJ=�����>ff&=�%c�(D�>��">ˡ�>wP��u�>)?�>�橾�I�=��=4Il��H�>y� >��N~��o�>��F>�c�>{0�=�����b�<9>d꽷] ?IM����ܽZG�>�ܽ �>
h�>�-J>˼=7��>Z�">-�����!������`�>�s=�o��D>��>J��o���=�m��@�����W�{k(?�ں�}���
����Ig���V>��D>fI�>z��[�P>o�J>"��;�T�;35��q=j>�K<����2۾'ڕ��1b�w�ɾ�p���$@��'�d�#>~�>�1���O!>�bN����>8�?�_���yZ���I�Ѯb?bg
���ȻxE0=���R�>\ a��L�>��>�܌�1%�=T:x>֐��	?e�`>�
��峌��aԾ A�>��u>���>sh=�9����=�L���g=�&��~��=�̊�=A�> A�>u<F�]��=
�����+?Mg�Z���F�C��c�4���>�{5��rh>C��<0�>?���?ǧ=$֢���F�4h0�q=ʾ�s�F?��>#���@��&S%��ǯ><NQ>�r"�q��=ms�>�{���g���DO�}�\>V��)��>�d ��!�nL>}�� 
��)��'ڵ>�F���]�j��}��eJ<�!%?������]>>�ѿ�������w��>xч��1�e� >�t3>vɾ	��=~�>O@�=� =]�%?��>9�C�]m=����W`>`X�<���>��?�w��g$?D�a>ap=􉜾
����A#�ѓ2==�ž��f��ʎ�8�=�;�����e��>�f�>i5D>�#>e��>�7{>Z�9>�mѽ#�=�E��rP=��+>���>|�I�i9P<R�S���2�"=£�>�A?rP��-����}`���>�^�>Z?�!=�$(���^�g�>��ԾGr�����ղU���}>��
����=�,?�ŽF_������;�? �?ٔ>����<	�6�5��O��u=>W&l�5)����=X9��!;����?��>���<�fE?%�����5��팾<����l�0ʾ�>�;�=�y���<�>��>�ٶ���c�~ >�N��~ɾ����y����>����D���-<��� �](�[����?�о��2?��U��k&>�lM��<'��>��:�Gɾ�s�/�2>�`
:+
>���W	n���=��<��>�ߑ>��=%̼�&޾+� Q��� ?%X�8�B>c�߼˄>8J��5c1�cz���㽻�G�?Ͼ���E�>N�1>�'/>��>�ﾞ^��	�E>3Ŝ=��žN�>"�R��Z=Kv�>D��>k���Ƥ>�)�I��>eS�ѮB�[�?=�L8� t �ɫ>a�4=�ϒ�6����q�>4�"?�Ue�I��I����=�٭=B>(>[B>�B>�r>+�G�=_^�����=���uY��,Ǽݬ�{��>��-@��Ҽ�=�>m�;�ȵA�]�>���2�b��;?='��=��;>�C�oG��	� >�w�=,���=�-0>ȵ�V� >3�h�!v>�/���77F>*��ph�%�ν�?�>�(>T�'��g�=%�H>��(�h\8>�yͽ�n˾\8�����Pp�>g�<Z�>��Ⱦ�o.���=]�k��ӈ��~?�+$=��c����>��̗��u"=�o9���[��߹=�ㄾb�H���U
>M�:=n���C祾̴�>1Bp�K;>2i�h$�=/
�3> Fh�󓊾�7��ܕ�� >@>"T)>���>{�˾�A>������!�>Q���m�����a�'�Žni��n�g>w�8�6�;��O���;h>&K��f%����(�;�O��Xʢ>��>q�>ٱQ>�K=>3m'�����>�8<��a��r2q=PĂ=/�=��n>
.ֽ�Z�
�(?��x>�=���DY<@٤>ݙ���+��K�>�gɾ����J�>�z[�.�=Z�>����[|�>(~���Tg>��@�佊>W	�=B&���&?Ttd>Ψ���w��e���GD<*t޽�n�>f�G<x�=���Nb���L�";mt�(���r�>Ƣ	>�����6����=~R��<e���?>rP�]���#6>E�>n4��D��x��<1����>!�`v�>�`ھo�H?��>R�>�fX>�p�>�V4�	^�H3�>�'��=�Fw>v��=h\?�r���b��:�%#�jj�=��(=���ڏt>佪��7x=t��v����>��=��c�>�湾�>�"�v��>����C�>��>g
��XV��1�>�р>r��=Uj־�|�����x�>f��=�o�*���S��#�3�ߦ�=~��>�D>+J�0/�>	3���lv�����uŽ���>�>d�.ɼ��Q��>����UK�,�߾���>m竾e��>H��WC�>�.�=W@=@M-��~�=X��=��P?	��l>��==D��A��"�V>� ��M_>�D��)`>�HC�M�M�޾9�.��-���=�r>^.Ҿ�.F�E�=p_=�M���1>=~o��� >vT�>L�!<����=��>Lm�=J��D�����=->=��=e�=�3'=�������4hh�[��=6v	�n�r=����O�U�� ��C������?@+��f���A�>z�>��ɽ����'���2?�ۛ��;A�����+�=G����� ����ʦ>���'f]�M-�>�䋾,Hþ����o��=I�2=T6l=σ{�$bj>fI >333��T����Y�D4Z>9���T�=�y�=WC��>"f����~r���V>�����2۽��>>�>�Ҫ>�kl��!<����n�A>l͖=��Ӽ$��G
>2�>U�>^K��/47��N��;>R~>�l> o>G >>Ϡa�}^�=y��o/�=��k�N���R�J>�j��䄉<����jM����=mV���� >���Ve�>��H�,�z����<�%��L�D>�>Q��=���=:]��t�<��>�Jþ�,`>�"�=�>��>�ǘ��|=�{�KYf��p�>�>@>��Y�j�Q>�c�<��S<���<S��=%̤�n���g\��=�^i�֋!�y�����K>%�=	��=Pp��,C��i�:9(A>xb������w_> W�=(>�E*����J�8fY��	��ȿ+5�>�"�=û\�2=!>t)������O�=0ע=�O=S�v>~x�2I�H�	>��Y�Ⱦ�=B{u<BC��y!��[�=�%�=�A��Q	���=�ǽe�>6���˾>���=�����y�>i:�>��Y��C=Or=i5�>�1�;z�x>�w�N~�LO�>���:$�T��d::\����7n����t{i>Z�=��N>�hw��'���>��{k��H0��)z>&S�>
�1>{NZ�-]A��Ԇ�X�,�����[Ž�;�U�=��b=d�>:O��f�j�׾�?����ۼ���%z?���Cɾ���>�Ǽ�[i�m�����=]n�� �>�q$�sǾ0a�����`>�Ǜ>��$>����B&ٽOv�=�h��2ɸ�Лj>�}@>^h�\�<)�=@ل>�#������½z�L�Q��>� =�žo�t��=�?D�?Vp��W�o/I�:#J>2Z'>��3>8I>���;��x>�Ԩ>��?\Ǹ<W!�>���>�ާ>���>aO�>�>����k	���m�Y���)$=��<�,c>�ʊ>"7��ϟ>n5k���#�����T>M�<�n�>k�=���RQ>�g�k�=�$>1�j>�7;��Z߾�->�a���Ͼ;��>f���Ʌ�@L¼(,1>s��=<NQ��'�� �*@w����ŽK��;�!�=�J���!��s�zSi?+��{=�[=��<�S��� ��w���r��?t�>���<�U�>���3�!�!>�W3=q�����>D�l���/�f��=RGG�g~��p�<��]>���=>"f>^�:>��-���Y�]�,��q=&6?>��
�s��>��>�cZ�~ �>��=�@s�WϹ��1>���8�b�6�G)���>qx=7����zA�~�>}\>y��=s>.t%="l>ؼ���=Qf�>�PK>�g�=x�� )?D��>���=�FL=����
0l=;+�\,�g
;A�>���=O;��
�r��د���>��v>~)>9EG>��ٻ�BJ������m7�$&������b��K�>j���>9���c����<=~���"�>'� �����６>��h�Y��3>K"{���>�B�PT�<j0ͽM��:f�~��u]�lx��u�==9�=ۧC��7�S�
���y<�S�=�_�{k�3�f��7^���G��!�>Ll�>:z>S"�1�~ ��}%>)v�=��|>�w|>�l�>�D�>��8<N�+�%;���)4>l
$=�	"=�"ξx`�=N>|7��M�nQ&�V�p=�]>��b��>P����=q�v�&�1>A�h�Ym�����;㌾Cƣ=�o�<�̌�F=x���R,w�֌�<T�)����᫽e���xz�=j��>EA>9(>[�����=�Q=C7��9�߽Oyt�ӽ=�`�>��=�Ag���.��)�;qm�p�<�t>����r��=����/�>�_^����>�&� �>}?U>��B>`"���`�=ݳ=4��>*�Y>�֝=�����Q1��0�>�7�J
�<���=M�<k�ý��]�{=E/�=ё>1(S��U�>]�P��K���>UP�=��7>IٽC9q>]ķ>b->��%G�\Z����<�D�=V�k>�J��E<�A)���4��>/��=@���+�|;<�c>j>�>
ѽ��>R
�=�Q�;�./=+��=M��>��>�5J>$b
=^t�$(6�����8[=I�1>�Z�W�I��s?��r=�?%���<\�<����4=��R>J)��A4<�C��[��<����<H�>g�'>�<��#��Y��<ff�>�L����<�ƾIc���_��ӥ?��E�>�Ⴞ�����3@C{��&�'=3��=��<�|����O�s�����?&����>>�0��{U�KX�<�7<!����=�ͽ�Pp��->�����~��+�;���m���J���m��,���2R�" �&����L�=Nѱ>yXH>��=RI�=�=$b��'�5�t�<;�=0/�B2��e����1|�>/��=.VԽTR�u�ޫֽk�b�⯙��8½�!�>г�=�D�����w½�v5>�=<��9>��t>Ve>�Z���U>L�6<\���Uh ��=>�C?�3?��Ƚ�˾�b9��_�=��ľ|~�>����<�>Ǻ�!���o5��H��E/c>��>�kF����=�����9>+�;k��w�>��"�[����}}>�X�� )b�r��=��= $�>?�V�3æ��l<�)Ч�#ظ<>y8��&�>�؂�A�(�X���} �A�>*F>���~7]�L�`<�ˁ:,.�ߍ�r�y�eS.>��м�L��{.�Oi<1?7=I� >d�����=/�J��Y��'��>���>B���K>ƿϽJ4=�%��\�<�(X�t{>���=QL�j^>��ټ�T>� ����?�&6�������¯�;�ɽ>Y����=��ݽ���꒑=`=�iWA>�@��,�=F|�:X�>�`N�W5>_�>�Ac>0�2<�;l���¼ϺƼ-�?=�t���{8����Dn=���雔=*o�>���t�����>�C��Eu=���=S��=*����D��o��>���>�O �5��=��ĽW�n���k>~�K>�c���Q�~t�>�v!>OZ�=�rB>�U(>v���v�>�=
�:�R�*>����۽�=uv>�c��=���W�#>t$�=��B��=����8���-���w��0�>��=>�D�e�ʽ��=@�x���� �=?�0>ѝ=Lp*=6��>:z|;�ξrm�_Aڽ��s�Þ�>�6���,�=����T�D��H�>�ļ�x��g~�=0�>��>uvr>F�=�Ӹ>|{=���>�K}>Bx�>�6?g�E9���>�'��۩�M-{���=�b>��;�_�=4�M��Џ>U�>�^="��>�S�����ҌE�b�1�Ӈ�=��>6s���A����U<�^������>��U>�����n�����>��8>�v=�8_>�	.�c�ž���n�=��c>od^>�g�;  @Ra��/�
>)�>&r���h�9�n>�M�E���?vTž�+��T�=���@�뾎鉼��J>���B>g+����.�F� ?���>���>� ���}>��=��T��%�佚>w !��ʧ>��=�N�>OX"�gi��B�>m�X��C����>�3>��\=���=�1=��=]�{L$=Nbн��>�Î=�|v�"�V��tR�5{�>�q>>�u �,+�>�h=��g<�� >uw������h>K\G=Ku���	�=�g<>]��>A��������>���=�=�ׄ����>�>l&?��(>^KȾ��,�\=�="�&=�?=?�'���"�ՕϾ����ٱ�=��?=w���5�w<֐�ף���?�>_��񺮾�L�<q��>�~
��(�>G�>���>��վ2^�1�����R�q���*P=@ڟ���>44>I������=\W=r����<'�=��J>�fU�q�e�� >�>�cl��%a>��r�󑴽n4 �� �;N�e�?��M>p|��>�&�T�;JA����V>q,��Q+=`�_=�v�>jM>��=qXZ�zS�KYf>����7���B��x�~��>R�
�3d�|a>*Ra=jjٽ�q>n�S�Ӈ��׆
>�P%�p�>���=-r>C煾(՞>S>V�K����=,���>�i>������;l���>Ƣ)���,�<�2�=������#=�CK=�q���R�G�>�E>�H�=l	?� A��8>�9��¥c=��@=v�6��ܫ�,� �ۧC>B�@>fI@>n�;>5^��]���i!���'>M��� $K��Ͼ'�>��ӾP��<���=TR�nk���e��kC�d���Sľ���=�l>�7[>���>��>R�S�1
�<Á���=x\�>����zh>�ڪ=s��<��<V�K>�56>E*>&�ҽ���<�R=<k��#־Tn�<Y��2䘽	À��=�>��=���?�E�=�mĽ�@k�t)>]�=��=���?���<�������o�0?�3�c�<f�8>]����r�)�>hw����ev>�q>��ݼ���>;S�-&��k`���i�-[k�t$�><f`��C�1%r���پް�=6��;��ʾqU��猨>{14����=�*�����_��<���vᇼ�ؼ>˜�<�sL��w߾���c���@?R$>���>=ё<���ۼ�Y��[�ٽ��b?5��������#>k�>J�O��)�<����%����>��Ǿ8Y<��j���^�.�>M2�I=Ҍ%>Q��������zn>v�ۊ==v߱��Fh<dv���[�%]�>�DU>Ƨ��"O����ľ%͟<���=�������=�?�9�R�½��m�	��ٙ��Ѿ}>���g�i�R=���=2w�>6g��/��>Y��>''?���>���=�?>�d>�2��0�_�>�3�����܂�����>9^�=R����&���׻2�2=�1�>��D>���>���>����#�>fk�>a�.�����= �>��@�X��>�:2��=�	L�����[��������4f���=�!�=��=��G �=
�G����<�'5>��=��=7 >��=�����5�=��_>�<���=��p������"��	���?�|��TR��zp���hǼ�&T�Tĩ=�n�=⫝�n��=��d>�[�=��O�d;�vlD>Ɗ�>}�B>ۖ=�K>C�>�!��>�GD<�<)>���j� �&6>�2����̼�O0���=��F>��!>q��=�����6���a='f�=��=MJ���HM��A�>�St>#�FҮ=�߼�̾"qO�U.T�T�ٽ���=�� >�3>M,0����S�Q�u��7�OZ�=�>����I0U�U��>r�.>(k>2�7��aD��Ę=8�\�%#�>��XW��ob�=�>4 �ղ5�\<��>��>|�0>	8d�S?/�/Q=>&a����.�W�����̸�=d]\���>d;_=�v�>�����O=Ϡ�� ���:��׆j����=
.��I>���=�_
>�p?�[>�	���?b٬=��L>+����)Q>�%>��@>��=,��>�)T>%yn=���>P��=����ڠ�yX���5<X��>��ξ6|�z��=��<�^�>� ?��=�:�xze���J>�B>�?�2�>���h�H?�}ּ�'�\>��	�=2�쾋��=y��M�ȽvO޽���>�a������,>�C��>�t1<QJ�<l&�6�!?�`w�ʉ�=�X>�5��"=)�[��4��c�=���=h�;7l{�l�@���=B�>֋!�O@��dj��`>_��x�w>q=?���������,�@���}�3>�������;��=z4�=�.�?1�ʾ�̽K<@�� ���`M��J�>w-�>���=�iľ]0>k��>N(d���">O�D=�oٽJ�T<�9O<��D�� u���=�aD>��?�F눾�
)>1a��@L�^�=�lH�L ���-;�n/>T�=n@>���J��>��� ��0��<��������c�qu�=d;߽u��=\l>�5?��}>��=��Խж�<{�ƾv�B>��=o�|>﬍>�]�T��<6�=�&�=�����>���>����M̾�&��z�ݽẄ�c�%��ģ>�E�=��>�򌾅�������l>i�\����w'>(
4�˹�>��>�� ��N=0!>'g(=��=Ӈ�>��p�x|�ın>�>�G{>��=a�^�1�>�C=��νS�p����>VEx=膦<4�>l�<B>�>�3>��X=Sy��Pp�=�~�=�q�>����m�>��Žs���%a��#�=|Z=M�<M2�>�b�9|��x����b��gW�Q�I>��	�%��Iɼ�?�:����=�ɽ	��>��>h?�>X�u> þ��н��A��o;��>�\�=�j�>����*:ҽnn>�2�7���1����H>��=�ɽ�|_��@�<��>��;����%�=qVD��Y¾�>Lq����:=�"��L�A���rv�t)?�����1�=�E??��y���R�=��-=��0=gQ"?���� ��C1���>�`>i�ݸ}T�ܺ;���>r�>Y�Q=�b� �g>V�>�s��e�>{�>fz�m�_����>,����t�>�:
�m�=jJ>�;���3��g_����ޓg�L|=�5m>�#��f�tA����=V�-?U0*?Ӈ>o�:�>nn��n:��M[=�3r�P�L>�j�<qm>���>����S	�<�1�=���������p�>��z=V[{�h�.�.�F�)?�>�<�<@��u> �>��[�'j�=t�=�F=��>�u
?��2���;���*?F����0?Ĳ�<ƅc>Ae>F���b�k>f��<Xs >&0�:���[��>d>L�&?%;v�.�[>b��>�#\>�> o�>�p �gaO��w�W�=>��Y�#��#��>n�g��8ܾ��>t�>��1�E�>`V���n�I�Q�)З>Z5=�T��c#�Ԛ�?�n��Z/>A}�>J$Q���?g�A�Z�M?.��?8��<l��>��u��ۀ?�׾�*>L���.�Y>	�k��l�>���>ka=�0���O>�>ľ��V>e3>�F7�N��=���n�A?�~�>,+���t!?�xܼz²�A�*���=~o3�=
7> pl��p>�M>h"l>�Z½�ٰ>�3˼�D���
,����{k�0��=�7����O>�:�Nb�>a�üH��=�	��d�P�Ӿ�ʤ�\rܾ���>���=oL>L�������5ʾ�?�.�RD�>�������>?�;;�=
w���&>`�?=
��m�?�>t��YA;Bx�>2�H����>�=ɽ<N�?�R��~>�����N�}v@�
�C>	P�>���l�\='��f*���\Zͽ�#_>M�����@ʾ�;&��i-�^��>x佅wQ?��>V�>겈>ᳵ=�������9��0�>u��>5�>3�����>R�6�J)�=�d���ϥ��pF���ᾛ����P>�i]<��(?�h���?�Q�� ��T�<�9ݽ�U���o?�Ҿ��>�/r����>�\=k�?������H�$��/�]�>='}>�ă�=�5����>S�>���>_F�>��"�d��Ǿz�}>�e���%l=f�L>`��=	����}�<dϞ<>�>� �>�{�����^�>�_��V�Ž�N<�8�<?��B��⃾S?�>(D�>G�����p>X�4�f�T��<>֋�>� ?{f	?)"��
ű�k`���'?�>ѽb�p��>a��>�� >B[�=�2[���Y�D���ݽe��<gHU��'B�ǀ�=Z������1|��V)���?)���vI>�ҾM�n>�t=sK+>�P ��(v����ŏ�6�=�cI�U
��Ϙ���=�	ڼ\ >����4��S�>�����$>n���>��m�nݽۧ�(�>s��>��>�4>�LS�l�&?4��l0��D?���y����E���ٽ����;��>��? ��>+�����>-?z߈��D���5>$b
>!��>l! ?d��>=�G�����4���v=>�c�]�=�V�= 䄼^c�>�B���q���E߾�/[>S����b��5ž`�>�^�+� �ྞ�$�1�\��!?CV��uZ>3ki�aTR�U/�=���=�o�	84?#=�;:@bgb�b&�;P�Ⱦj�=��=Y>Ho8=���?�`a����H����{�|�>��.>L�{�h",=�b�>䃾>
ܪ>k��v�0��>_�x�]��>h���V�>[|��� �sK?��c=r�t=�=K<�>���>s.%�̗�>�b�>���>x��
����v�о^L�Xś=��6�?�͗�h�3>~�:���=lu�u��`��>�����q��:�����ȽD�H=O@ӽ����Fz>H��=DQ@�*�>⨾�U-��*a>�Z?N��EY>Á>#�??�혾BC?%>ҩ��o9��&ҾO�u=-�	���>�+���ξs������d�
�>Ѯ��c?F|G���>Uܘ=Ӡ>K˨��c�nQf�TR���l�;R'�>M1�=[��;��⾔ۖ=2X�=���=0!��� ������ֽ��u>�%>3mg��̯�K�ǽ5)�>J����� ��C׽?&?�e��"���wv=�N�>�ĭ�X�G>�ux�;ߟ�L?��>�é>4�=��>���>qU���N=W��>]��H�鷿���t>r3�>� U���|=	��>�V>�I�<�����)?�>�>���6�����`v�>���>C��E�x����
?� d>�¥�i�>�ݾl����=�����O?^+a�?��>�[�>���mV�PSK>�˼�o;m�������5���ڽxч�y�5���>�7�>�j?�uVK=��=�฽e�۾6=�=
�=\V�<~W�<N�>L�=Hmb��C��,e��5A���g�7�ؾ(��H4��I�w>��L�Q���>?�?�k>N�}=�4>��(��2�b��%�>E�>��I=�d�=��?>p_'>��*>:��>��P�� ?�N0?��7=8g� {=>Ih�����M�>���>��>5{ ���F�.�;?4�����=�o�.��> :?[(?C��>��=̗/?��=y������>Xν�B��'��>t)�<�1#���!�и����T��'[=峬>�P待�����M�޾�׶=g~��V����8>��+��WH�7�.=!��?������¾���L���K�>`Y���w�3��=�1�,�>ҩ��ɐ��<r>���>�I�<O|='�:���<���ʾ���>N+��������>�e��M�>5)�����@�籾NE�>b�=�쵾;pN>�K�������?9�E>�lɽ0�s����=�%�<��V>k`+��` �������оA*�.�u>�<	��H�7l�>N��>�q_:�;�=�
r>�Z>���>�W^���ʾv��>��>��I-�=�J3> ��=p_��5���˹�=��PjԺbܭ���>׆�>��-�-쩾H�����0:=�9)>/�!�W	澥��>��>��K����>[yɼ�!���O
�V�<���=������=${Ľ%u����n>fKּij���x>؞Y�K�8�Lq>�̮�0�A���]>ؚ�}?5?��>�ǽ���F���"����C��n���n�=�U{=ʰ��`��YQ�ػ�����lJ����B[.>C#���d>����;�=p|-="l?r�پal!>
������=�3�=�&�Áн��?l��>�1 ��^T?��.>�}�l&�>ȵ��y���	��=NЦ��i���>�����>��d�wҽr�>�����b�>�,�=��=ı��^���>rP�f꾋���>��>�=�o�>�/��*L>�X>�Zļ��ѽ�׼�w�<$���;=��$+?=3����}>H�=>��R��>+<���u>��^���aq?@�����I��!Z>�����k>�����<���'��>�g>F�3>iRr?�u��Z�	��pw���:[���>n��H�$?�=z��>�-m��T�>T�ý���=�s�nL>�>M�>�Q��>�>�o��=�dʾ���u�A>r�=ް���i>�?�>��=���ؠ=ӼC���e>��Y����%A�����q�>Z�����M=��W�J�E��^��-`¾1_����>�K:��+���P<|~�� �S��Ѫ�m7�=�������>�|�=͒��_�>��/�[����h=A�Q>�8�=Id���G�> X���S=�=�@y=�δ����="7>�3B>$�>�F >��u;}����=p>�y�>�2>\8�>�F?�V��ڝ��4>C<�����>�[ ��"�tI��6v=�.����C�r���
L>�z�>j�w��v��*�>PǓ>���=�y�=!�>sh�>.��ݾ��Ҽ c�|���d~����(��������
�>�޾:#�>L�?J��>�*��`���(>uv�>�hW�֐��ž��>9�@wg�>B>���l*�/Q���v8>y�>w����3;?��[>������ľްm>�	�=��&>�$�>�ԥ��H>V+�>	~�n�W���о�p>T��%#罯�w<.⋾�"5����� ^W�����O?�1>���=�-�;��>��e>em�=��<��׉�hvݼ���=,H3�`vo�&�W��h���'�ĽI���1>��6���l=��w>��">�:�>��֋���
���7��v��ˡe>�]�|~�F�H>*R�>*�#�-�>��ڽ"T	�(aƾ�����E��\�>�|ۼ�%'>�d�~t�>�]9?/�7?:O�;�Yû�;>P��>�A ����apM=2r��v�����j>B$C���?����>TV��.�=-�c>�
�=N����=}Ν����>`�����>!���|���"?w;>b���Ӿ��k����ϖ�A+���
��W.��dJ�"Oҽ����κ�r�Z��<BC��a{���>ZG�>|'&>�G���k��J��>��b�gѻ�*SL��C��_�,����IK�>�=�U���=��	�"?4��$����=�%=�t6>O;�>1�>��~>H��>��4�0c���J���>�~�CV�=�s>�}f>�վ���>rm�>�3n�v2���(k����&��<�!���1�>���;Ӥ>x�G�uYl���z����[��¾��M=6X��M>�Jm��>�1�=�x߾�ξ�=̾�}���R�j�T��gH=�<�<��;��=D=�Ș#?� (���<V+��ɪ�=�->-�x_��<k'���N���������c>���>N��>H���0d5��3>ڑ�=��������2$>�:�>��>�<�G<g�Q=��=>�j�=<�=-�<*R>��E>��=&�C�`�>�O���o>x�2����'�>.�l�j� >�uZ����(�����ᾺI,>�N��W=1徦�H?r��=G��w��=Nk>�W�>�P�������>�Kǽ|�
>�@�����>U4V=�p��y:��>���Z>�z<B	3�q��=�L���>i:;�Z��>���;�?�<,H�I.?�>#J[��|����c�al�������S�>ⰴ<�+P=�� >����]̼5����ж���4>%#�>��Y���=�x�ໍ��p�<�ֽ� j>��N;��F>H����ܯ����=� @wg���B���=��-�����
��=�D��e~?�=�ߋ==���`�=����>��Ƽ�����Y>E]��m>�խ��+`=��%>��8<���>q�>N�=���=:��>DQ�J������>&%���[��>	8�4��=�U(>R���<[�Ad��tB(=7O�>�
r>#Y�ϒ<x�=I����7�>�}�����<(*����J>* ����&>�x��b���X���$I�x<>Z�> $�>׈ =j��;���=�3�=�gܽ$�����y;�>*c�PS+��Eühy޽�1����S?���=�	�>g'�>������>���C9q>�:>��2>�ﭾP�(��>��I>�~��˅J�E/#��J����<>W�ǿp\Ƽ�I=��`���q>�PM=p|�����s�@>~oӽ�ȶ>ގp�@j�z<і󽙷�U�����h%>��}>㪲�����ˡ��З�?t��~�=�������C�>��=R�=�,�C�O>�Ȍ�jܻ=j���rV>g~>�@�>p�3� :��D��=�؏��>>?�
>�Q�W!E��%�=� X= t8��75��Ub�"7#>�=�=�W3�Ǻ��h>���<Ll��?���\��X���I�������bJ�>Toͽ������>��	>�rY=�U���� �-C��ۖ>�g�;j����>/���Է���#�b�>m盾*:R�%u"��>z>M���0��v>֪�=q���w�#�>��?������P���w�=��ž��=ڏԽYn)�;��>�|>G=$�'����ŵ<�:�m��=@j�>9���׋>�/+��1�s�>v��>��=}�>��>\=>���n��<�
&�m�Ծ����>��.:���=��˽^��=�-'�ۧ>xЬ�
��>y�f���>��g=��> �0/=��H>@j�l<�Hb���=!Y�:�L=�:�>@��,�j�,}���ƽ�K�=�5*>�t�+��>�b_>	��=���>��>Ll������n6?��U>Uj���`Z>)y>z�/�.�&��d>���!k�nN�=��>���>=���	�>0Z�r�)�~W$�d9�#2���s��ҡ��X�Ǹb=t�=	�>&�}=��>�4?�5�+�~�4=�Ph>���=P�>U���lm�R�<��9=��V��q�_��?�@@ެ��EK��'f�>���=�թ�r�)�1���H�?�ؾ^cw�[R>֏�<du�=p�ɼ�Qս �I>�N�����X�ף��$�%��b>"�+��ڃ=���=���c��S��
�VĽ�ŉ<[�>�N$=d@V�(~l>�t��KľY�Ͻ�6νv��=������=j���aF>�Pt>�W�N�>^�	�a�Ľi��ƥ�=�񣻕�����>B�M�e��4�<�x>d)=���=��0�`˫<�|�>��� oA����Խ"�P�8�P�a��>�R�=�b>ػ����`��_>����������>U=����!��;����>ni>� T=�C��˟����<��A5=I�q=u:>�:�����u�=�z����^���>�+���d�?����A>���֋��"�=����%�o,h=��.>	�̸�=�
o����<��ܽ�A��;R�=ѭW�R~*� ��=�B<9���Em=czB>�*=�=��0>�t�<�X�� <>
��u���؅>�q>��Q>�����r+��Q���2�Na%��w<����_�i�<�v����=d������ �O�BC��$���^�Mgg>�#���d���3���,�M�h>�+];s.e>w�߽�I�=L���z��=����l��,H�=Z�=%;���<�:���ZI�ÚJ��ӽ
��mq�H���Z��=�92=�s���_>�o�=�K���վ�LU>��>�-�=M-;�����V�8�>�?;�_�G��4ʽi����C>Nb�=gՇ=�ֆ=�M仭Q��nLϽ|��*5�OZ��~Rm>�tS>#-���C��Gi�󫉾���<��=獳�>�T>*�y>�z�=L�<>�~+�c�?>����qU>%!��$�<�����o�<��'>�H?�.����>�>�=�U�;�O>��>��þ��<2վ��n�5^Z>���=AH��A���|�$:�Nc�G8>�t3>�2�=�\�>pA��	�>�I=隹�pC�=h��>|'>�q��4>it??�mZ>k+>����7�����|�����%�(�� �>BC�>��+>N>!�=m�=q�-�t���F���e>]�Q==IZ>m9��|~x����=��_<�A潯�����H������H��z>�6>�H�>+~���@�[B�<����,�;��vT5>W5>�%@���_F1����>�L۽T�>��L�'�C���?��%�tq��p;N(D��o��!� =j���f,�-�C�GY?<�%���`>:��:�!�>�u����>��>D5�<L�;���ջ������\V�<�Ƙ>~tj��Oj�%;>�C������fڻ�?X�~ >��>t^c=�]%��3�=U���d�L��'�6#C���*�F%u�RM�;�kI���Z<Q��>�ν!�w<S$��k�	�F��=�9�=&�W>J�T�k�=>o�6>�")=�k�����V�� ����9��ض=�f�>4Kb�\rܻ��u>ym���b����=��F�:�X<D��\w�?�<L`<.ť>��!=�,=����i:�>���=b'>�HϾі3��&�xbV>�'G�Gr�>i o>�{>��Ǿ�pY�@݀=Sz�=�e�B��=Z��=�4��]��>����w�<P��=�&n���޾X���7���.�@��白�c@��	v��^�<����\5��O
�x�=�սr�>�r�;��>��=��¾�⏿,�j�YL,�al!=&S�>��ü��7>`�
>�[>��W=A+��?���c>d�=ɓ���=v7����K>T���q�'�L�0�vT�>� !�od^=��y=Y�Ⱦ���>x�l=aO�>��<+�|>�_��� >b�V�{Ic��V=K:��N=t��Di/���{Z���P�'Nn�7��=@��=���3�����e�.���e���9E�]P߾�R>�<yX�>���=[>��߾��>�A�ʦ|>�ǌ�i�">���>��=�|>��=S�M�|�=B=��1>Ƨ ��̰��P1��O�>Llξ4������&�a����=4�/�tA�����{fi�Hm�j�t>i��>��>XV�=C�y��3˼�
���7>�Z�����+0佅Ϋ��e�X�=��ܽ>���=�*��7�=�����=A}+������j�}$�<t�Ἆ;%>A�=��F=�v?��S>��>x�(>r�����=���=�q=�m���v>�����F=�ng�c��>='ݾ��>?T�����y�����)>�~;��鉽=a�>E�)<q�>�΄>�V/>���戽��<�����2=��0��S�>��I��/۾�-><�f>;6��ΈҽՔ����=F>�;��y><1+>ۢ�>��(���Vf>X��=iǍ=}̧�x�=<�6@���Xm=��Y>�?�=2���j3μ�i4;�/�?��C�9�����=�9>�2=K��=J���[~�
�Ҽ)��8>�`ڽ���;(I�>�B�=Է,�����3��->�k>JBb�I-T�*t��~q�=�%c=��<n`>,�>:�������=��L�c^�<�J9>�M�=6�O>�q��^�S��rQ�y6���=�k�#�������-�>ꕂ��"\<x�=�(�=Xuּ}e>���>���=@�?[Ӽ����>K�=@��=�=���������>��>�ː=5���S����w�E��ɇ=[����>M5>V1��s�S��������>�Rj<K�]�����ޓ�=�^>���T�;AνP���σ�>�"��Ж=
,���s����뼋8�:s����D}=%�[=�Դ; �=�#��(3>Ŭw�a�s��\=�W0�4�4<O�����=�6p��J��9b��">�r�0Z>jͽa.������s�>�0���� =�ob�=���л��޿��z>��u>��˼�`>�cc=�rn�LM=&:��i���N�o�*�p�B>�_3�O;��\��H�>?���Y���W'����a�Gu�<�o�=r����ʽ�i�<z8A��Ź��m�>>�Y=�s=d�Q>Hé<�kL�����{I�>4�z�ܺ�=���>��?��>��=�
6�?�&=�d��7��8y����>���< �S>h�>=pž#M<�V�T=rn�=Է�<��R=�컼,��>�u��{��=�z��u�S�vTU>��=�5�<��^��bܽ͏���u����1��)�=~O�;U��}I�峜����O#-�/>�{>$��s��=Kȇ<R>��<b�w�әŻ�G!�`<�C����R>�`>�ĭ<��T>��n>*:�>g^���;�<�r>=`^=9��>J�e��E�>��=ж��:t:=�e=J$Q�6�o>�1:>w֎�� >��7=\8�(��>eS=�~��=�=��=��n>q�<Q:��]E>k+6����=�(�>�8�=E?��R>˾��1>�����Ǿ�M�z�O>:�6=Ź<�b�.V�>�&�>��=�W��x�?��=�a�M�>�=�N= ����̾=|�<w�*=si���(:�W���`��.u��,>��<.9�~�=��Q��զ�W��;vO>��0�=I�;�z��O@RՄ=Sy�=���>I��=�z���b�7l�ʉV?�������>�ܫ=�p`>~��=�����%�:X�S"��r�@>�\����/�¥�=uYl����=r�E�x&t=uɘ==
�=�<�
=lU�c�Z>�ǵ=�����y>Á����ƾ2�<�kE�<_A:>Tt��+ý6Yc�Wv�<��>���Q�[� ��e��.�>�>�=�dp�T7�=}\�>�V߼�Ø>io�=�8T=2w�>�h<;"7>Sy�>W!?�} �a�^>��/�B�>F눾� G>�]�>�3���>�=�|ܾi�<�!�=�P��>�D���ɾ��b>:;9>!Y ����h��<iR*>y|>-&�=}��=����Yi2�x&����ӽ=Dc>t젽�Ѳ��|=����I�a<�w?���
�\��>T�=�U߽�*>���>p�½P�l>��R��̒����<�M�>�?ʽ�ֶ=�A���<><����>��
���������> )>�g����=*�d>9�=�ޤ���2(�;�)+=c�{N:>T5�>x�7��҇>}�M�)��RI=>2u��2��zlK<�x���$=���=�3��潧�þ�i~��tͽ�=�@'=��B<K��=�=��$�Vd��%@M>����!q>���=�u�=�<���<���g&X����<���k�>^c�=1%r�p�t=$�>�hZ�Yn�9����=��F��S$�!ȑ>�s>?��'<o����T�=�ձ8���<U��񄞽2�>��>�!Z>����>�����>�@�<1�V�.�˽ ��>��I>�C.>��̼o�d�DT�	�%>r�n����=U*��:���=��5>낽���`7��P=/n#>��7�Pǃ>�AF��k$��
F����'�U>֋�=U0*>NM];S^+=]�v#��OM=���n�=��
�7>[\=겘��'��]m%��|־Sy�>cn�� ����w�'�*��f�= ��=$ؽ�I>*t޽P�=�!�>" >^���S�M>*R�>g~�>s�<0�'>I��>��?�a;>VN��7#��;��A�������b��`�>��>rm�>Z�Y>0g6=gӑ=x=e�@���<�H>�/v�y]?>�L�=����4V�4iӼȔ=�C��eq�q���*�>��8���=�(��9E׾�t�����=Sͬ��{s��3>�:>W[9@��.=/Q}�0�|>�����=ݷ�y���?�4���O��������F �4�ڽ�X�\剽����4����>���<Q����>�3��	��sK�>S����P�a�B=��C<a�>𧖾5A�=�x*�
<=�r%>��
=E�>�#g�>�ZS>�< $��.�<��|��L��}:��=����>{f����>l�F>\ !�H���Xū>�) >�-{=v28>p��>�¶=]��;�׮>&p���f?�P�=��>��=�.>V+s>H7�=�y>����yΖ=���e��T�n=�Cn�U��	ě������XW��),�}͒�Ǆ�<�4=[�K>_�R>���=='�� �ý(�$�fZ�(,�>s��<�^���_�>��̿O��p�g�]o��V�>�6�>fK�a��;� >���ƽ��9�Ral�3P>������=��>m���C�/��y�)��>�.R�K��=�F���>Hm>��;�b�=;U���i���>�>X�HQ'���F<'�C>���� �=��F>ap�=b�κO��<:>b��=��E�N�0��">�rǽ_�i=�$#�[�p�\�ν�w��;��=&�½|��£-�)\�>�����=�0���*��JY>h�
��Z;=_Ҙ�Ͻ<�?!>���>@�D=�>o��<��6�����$��>+�ֽu��H�)=�`������w�~t*��i�nQƾ-&�>��>���>a�m�P�ؽ�7����f��>܍��+X��50�=~�=Z�'���}���=1{Y�mŮ>�m�=�(���N˽��T=�^ҽ[|�>��{>�8b��)��Ǫ���=�̰�!Y ��1��z��1>��0>@�� c.>��u>�ɹ���g����=�u}�q=j�S�6���&�(�>�CV�>eߕ>� �7��>9���£->#��=I��k��=&S��=I�>�ˋ>�F���7��Og���N��3%?�q!>�(�� Aq�g��<.�;��)?:�=�9>c�)=a��>~ ҽj�E>(~���g>��ݘ�>�!�>��$>E>?������
1��Eӽ�H˾A׾<32H<W>+>;��>S?�p%�=�!=�ʋ>#2����%� )�>V�e��e�>VG����>�M<>�M����g�=�o�F�B��,&�g����p��u��>=~/�ms�>Z�=
�����>�L�=X�[��ֽ6�>�V	��@rP��-�=�f[>��b>�2��&�ǽ&SE�$(�?wg��ӤԽ��`>ղ�=����>�l*��b_>L�;tb����p��Ň����<ɰ*>ݷ�wgM>ݡ>�����=�}�>�w>�6:��E�K:�<V,���������<%U�;���=���=�����˦= �ʽ��c����=Q���L�>���=����d;�=���u=Q=O� >��7>���W�	=�IӼ�>d��k��%��*d>��]��L=<B>�>Uه>Ȟ�G��>�0��{�<!��rmH>�y�Yi�=F����㾲��='N�=�O�F|G��2i=��?�=���`=�J�=Օ�>o���=7j��I��=\�ɽD�>�G>r�	>r�ʽ�*%=�)��ն>3�>�q>>8I�=�=�����Xw>F�6>E�=��K�X>!��������=5�>�ڽy����"�BZ#=(=�$)��%>�$y=�p�>[e�ղ>B>�����>�Q��x�?�!=���P�x��ː=������^��м�9����2>_�=��i�=g)���`>���>��=EG>G�?�H�	��~���/�j>�7�����;�н���=B��$��V#>/�>f�ý�2�>k(5<ٳg=�1�<i O>��9�?/�j�T�>b�1>o*�>����p=�wo<�_0����=����iD����=�ق�ڮP<�:��]v<X��>X���W�x��d½���<b>�V,>�E��X4�䢚=t�A>� >:�K��$�j�W�L�>�H7=3k�=�><N(d>��1���E��Z��x���.�{�B!�>Ef�=I�>(����� �Q�=���,D=���F��	>�+�؞>����Ժ=�ꍾ���<FӉ>�=�>	��=�������V}��"��&�;�
?GZ*�"�>��=R��>Y�ľɥ���������+>3��>\�=�����n�=l�>�T�>MG>��U��S����n�Y�O��>�1=�=p%�>�o�O!����7=��>Vӵ=p_?�A�����=��>U��<h�'�|=E�>Ҏ��zpW>C9Q=�qD>���=c�Z>z���_�r�EH=Toͽ�;>B�<m��	�>�iA><��uڽ�%����6�6v�=�K�=�:�>��>��2?r6>��v>�C�;���>�*
��g𾾫T���?F�Q��<N��D���<k�?}]��D��>H��?y�����>tȾ7�?�:>�?'���=�������Y�?���nL�=��8>�nϾ�%>�b��ٱ?��a�x+>�a>�bgj��>���K�>:X�<�����@�>���@�>y�5=��0����n�z�?��o=��6>O@�\8p>%;~?颽F��=��>ػ߾��p>�D0�3d> �߼���>�/==W&|�0�>|D�^���|\>@����3�>\q�=�V������s�>:���7O�>�އ��_�NE�>�_��*��U#��jX?~A��y�����>�m>p��<σ�=+-���Z��#<��S��؇�9(�>��]��N���[�>|'־U���j�ɿ��1?RaL>�B%�j��)��іS>)���[|����f=N(�4��>�>�>n�=l[D?~�>N�K�	8>���>��>E/c>��>��>Gw���G�V&��,�=�{ž��i>F~��_��=���]ħ����>��>P��8->��>��p��۽a�>/>9b-�����O����4�}y1?��~>�E_�u<�>E�>̗7�k��L>C�?�e�>l�=�n ���3?k�>"�(?kH|>N�b����l	�>�7���=>�g0>ȵ��8��>j��v7�>K<@>_�>�y��W�=�{�=}�>x�=>Ƣ����{��4?��n4�=4.�������u��<���m�=~i?���S?/���?9��>��v=,eY��N�S�M>�T\����>�p��?>[�=e�?�rX�	Q���k=�Ξ>̡��^�؞)?^T>ߥ�<2Uо�ࢾN⽺U����>W�u꾝F���>f�^��,=�z�G��.V=niž
.�=��9>�WC��9�Q1��2=�[�Yi���">嚢=+��AH&?��j���޾z俾�8���/?Ȕ���E>��D><���3��>3�=3�Q>'څ��g�|,}�n�"����>�8�>J�;�^����>����J� �پ�(�=2w���Q�öe�/Q��Ii�<��t>�oE>[3<|a�=$�?xz=����0��>���>]�˽nn�>���܀>��v=�;�>/��>��:=�t���&��X �����&�>ı>��>,�=�WS���齷]���6���-н]�>^�>ڏԽ��M@[_$?�]>Z?J8�z6+��״�u�ؽ�b?��(���m�vO����x=^g�=[�=/�^����b!=M󎽨5ͽ��ղu��Y�=�t�����=@�:�}����V�����d�;=�/��UM�>#N�=C9q>�^�>�Z�y���`��g�>~}�n�'���;����
�d<���=��c�-��:�Q�D�R�@0�>��>;(�82Ͻ�R�>��T�?ё�>"P}=��>���=m��=%�$??�$?`>��?"O*���d>ƿ�=�Є=�Ff>N=�$I>n�r��־��M�����K�<��墾��>/��w�e�(F=�t<cb?
�N�)\�>�Ǭ=rkR�.s��1%r��	?uvr�T���[(>C��h���I�q��m�>��">ڍ>�"����T~>�v����㈥��#B>�K�>����4�=)?i>(;>R>�����/�<OX���>/z�_)˾�b����>��P>�W���۽A�?�.B�M�=�`�=���=�l�>�L�����=BxT>�J�����>S��S\�#�=Q�a>��b���<����T>L�p��Z�s+�������ϕ>@j�>�h;��=����u`��վ8羏Sľ9EG�F�9���>6|>܀�>�:�>��`>��U>�ܾt)����=���>�Y�=��*�2�"��=^.Ҿ�k=7l�����ӽ�1`>�G	?g~�>�x?̑>2UP�LY=��g�*@=�R�>"Ƚ�X���K���V>V}���x�>���!�>xa����<u��=@i�=��_�>��=����命`Y�=@0���mм��Q�*�Y<�8=����e>�����o>@MM�cє>$���)�,��/o�Q���f���2%��
>�.��9�>������ݵ���3>�.�=F��>N��a�Y>*R��ހ>JA�B`վ�Z���O�>M�����Y>Nz�,��=Ǻ>����]��>'��>#���K�>�`�>�	<>??�HP�>��=��?ף?���>Ӌ;f�>�m_�r��=S��]��'���b����J��<ƣ���K>Y��>��?�E��<��a�'>�=�=����W>.��5A�>)�C�W��;Ou ?;>(
4>��C>d]ܽa2U�/�>(D��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                *
dtype0*
_output_shapes
:	B�
Z
Variable
VariableV2*
shape:	B�*
dtype0*
_output_shapes
:	B�
�
Variable/AssignAssignVariableVariable/initial_value*
T0*
_class
loc:@Variable*
_output_shapes
:	B�
j
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
:	B�
v
embedding_lookup_1/axisConst*
_class
loc:@Variable*
value	B : *
dtype0*
_output_shapes
: 
�
embedding_lookup_1GatherV2Variable/readstring_to_index_Lookupembedding_lookup_1/axis*
Taxis0*
Tindices0	*
Tparams0*
_class
loc:@Variable*5
_output_shapes#
!:�������������������
{
embedding_lookup_1/IdentityIdentityembedding_lookup_1*
T0*5
_output_shapes#
!:�������������������
V
concat/axisConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
concatConcatV2embedding_lookup_1/Identity	Reshape_2concat/axis*
T0*
N*5
_output_shapes#
!:�������������������
f
dropout_1/IdentityIdentityconcat*
T0*5
_output_shapes#
!:�������������������
c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
z
	transpose	Transposedropout_1/Identitytranspose/perm*
T0*5
_output_shapes#
!:�������������������
�
7lstm_fused_cell/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@lstm_fused_cell/kernel*
valueB"�  �  *
dtype0*
_output_shapes
:
�
5lstm_fused_cell/kernel/Initializer/random_uniform/minConst*)
_class
loc:@lstm_fused_cell/kernel*
valueB
 *��*
dtype0*
_output_shapes
: 
�
5lstm_fused_cell/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@lstm_fused_cell/kernel*
valueB
 *�=*
dtype0*
_output_shapes
: 
�
?lstm_fused_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform7lstm_fused_cell/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@lstm_fused_cell/kernel*
dtype0* 
_output_shapes
:
��
�
5lstm_fused_cell/kernel/Initializer/random_uniform/subSub5lstm_fused_cell/kernel/Initializer/random_uniform/max5lstm_fused_cell/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@lstm_fused_cell/kernel*
_output_shapes
: 
�
5lstm_fused_cell/kernel/Initializer/random_uniform/mulMul?lstm_fused_cell/kernel/Initializer/random_uniform/RandomUniform5lstm_fused_cell/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@lstm_fused_cell/kernel* 
_output_shapes
:
��
�
1lstm_fused_cell/kernel/Initializer/random_uniformAdd5lstm_fused_cell/kernel/Initializer/random_uniform/mul5lstm_fused_cell/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@lstm_fused_cell/kernel* 
_output_shapes
:
��
�
lstm_fused_cell/kernel
VariableV2*
shape:
��*)
_class
loc:@lstm_fused_cell/kernel*
dtype0* 
_output_shapes
:
��
�
lstm_fused_cell/kernel/AssignAssignlstm_fused_cell/kernel1lstm_fused_cell/kernel/Initializer/random_uniform*
T0*)
_class
loc:@lstm_fused_cell/kernel* 
_output_shapes
:
��
�
lstm_fused_cell/kernel/readIdentitylstm_fused_cell/kernel*
T0*)
_class
loc:@lstm_fused_cell/kernel* 
_output_shapes
:
��
�
&lstm_fused_cell/bias/Initializer/ConstConst*'
_class
loc:@lstm_fused_cell/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
lstm_fused_cell/bias
VariableV2*
shape:�*'
_class
loc:@lstm_fused_cell/bias*
dtype0*
_output_shapes	
:�
�
lstm_fused_cell/bias/AssignAssignlstm_fused_cell/bias&lstm_fused_cell/bias/Initializer/Const*
T0*'
_class
loc:@lstm_fused_cell/bias*
_output_shapes	
:�
�
lstm_fused_cell/bias/readIdentitylstm_fused_cell/bias*
T0*'
_class
loc:@lstm_fused_cell/bias*
_output_shapes	
:�
N
lstm_fused_cell/ShapeShape	transpose*
T0*
_output_shapes
:
m
#lstm_fused_cell/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%lstm_fused_cell/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%lstm_fused_cell/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
lstm_fused_cell/strided_sliceStridedSlicelstm_fused_cell/Shape#lstm_fused_cell/strided_slice/stack%lstm_fused_cell/strided_slice/stack_1%lstm_fused_cell/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
P
lstm_fused_cell/Shape_1Shape	transpose*
T0*
_output_shapes
:
o
%lstm_fused_cell/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'lstm_fused_cell/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'lstm_fused_cell/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
lstm_fused_cell/strided_slice_1StridedSlicelstm_fused_cell/Shape_1%lstm_fused_cell/strided_slice_1/stack'lstm_fused_cell/strided_slice_1/stack_1'lstm_fused_cell/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
Y
lstm_fused_cell/stack/1Const*
value	B :d*
dtype0*
_output_shapes
: 
�
lstm_fused_cell/stackPacklstm_fused_cell/strided_slicelstm_fused_cell/stack/1*
T0*
N*
_output_shapes
:
`
lstm_fused_cell/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
lstm_fused_cell/zerosFilllstm_fused_cell/stacklstm_fused_cell/zeros/Const*
T0*'
_output_shapes
:���������d
P
lstm_fused_cell/Shape_2Shape	transpose*
T0*
_output_shapes
:
o
%lstm_fused_cell/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'lstm_fused_cell/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'lstm_fused_cell/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
lstm_fused_cell/strided_slice_2StridedSlicelstm_fused_cell/Shape_2%lstm_fused_cell/strided_slice_2/stack'lstm_fused_cell/strided_slice_2/stack_1'lstm_fused_cell/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
d
lstm_fused_cell/zeros_1Const*
valueBd*    *
dtype0*
_output_shapes
:d
_
lstm_fused_cell/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Z
lstm_fused_cell/MaxMaxnwordslstm_fused_cell/Const*
T0*
_output_shapes
: 
d
lstm_fused_cell/ToInt64Castlstm_fused_cell/Max*

SrcT0*
_output_shapes
: *

DstT0	
�
lstm_fused_cell/BlockLSTM	BlockLSTMlstm_fused_cell/ToInt64	transposelstm_fused_cell/zeroslstm_fused_cell/zeroslstm_fused_cell/kernel/readlstm_fused_cell/zeros_1lstm_fused_cell/zeros_1lstm_fused_cell/zeros_1lstm_fused_cell/bias/read*
	cell_clip%  ��*
T0*�
_output_shapes�
�:������������������d:������������������d:������������������d:������������������d:������������������d:������������������d:������������������d
d
"lstm_fused_cell/SequenceMask/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
f
$lstm_fused_cell/SequenceMask/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
"lstm_fused_cell/SequenceMask/RangeRange"lstm_fused_cell/SequenceMask/Constlstm_fused_cell/strided_slice_1$lstm_fused_cell/SequenceMask/Const_1*#
_output_shapes
:���������
v
+lstm_fused_cell/SequenceMask/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
'lstm_fused_cell/SequenceMask/ExpandDims
ExpandDimsnwords+lstm_fused_cell/SequenceMask/ExpandDims/dim*
T0*'
_output_shapes
:���������
�
!lstm_fused_cell/SequenceMask/CastCast'lstm_fused_cell/SequenceMask/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
!lstm_fused_cell/SequenceMask/LessLess"lstm_fused_cell/SequenceMask/Range!lstm_fused_cell/SequenceMask/Cast*
T0*0
_output_shapes
:������������������
�
#lstm_fused_cell/SequenceMask/Cast_1Cast!lstm_fused_cell/SequenceMask/Less*

SrcT0
*0
_output_shapes
:������������������*

DstT0
o
lstm_fused_cell/transpose/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
lstm_fused_cell/transpose	Transpose#lstm_fused_cell/SequenceMask/Cast_1lstm_fused_cell/transpose/perm*
T0*0
_output_shapes
:������������������
q
lstm_fused_cell/ExpandDims/dimConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
lstm_fused_cell/ExpandDims
ExpandDimslstm_fused_cell/transposelstm_fused_cell/ExpandDims/dim*
T0*4
_output_shapes"
 :������������������
s
lstm_fused_cell/Tile/multiplesConst*!
valueB"      d   *
dtype0*
_output_shapes
:
�
lstm_fused_cell/TileTilelstm_fused_cell/ExpandDimslstm_fused_cell/Tile/multiples*
T0*4
_output_shapes"
 :������������������d
�
lstm_fused_cell/mulMullstm_fused_cell/BlockLSTM:6lstm_fused_cell/Tile*
T0*4
_output_shapes"
 :������������������d
j
 lstm_fused_cell/ExpandDims_1/dimConst*
valueB: *
dtype0*
_output_shapes
:
�
lstm_fused_cell/ExpandDims_1
ExpandDimslstm_fused_cell/zeros lstm_fused_cell/ExpandDims_1/dim*
T0*+
_output_shapes
:���������d
]
lstm_fused_cell/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lstm_fused_cell/concatConcatV2lstm_fused_cell/ExpandDims_1lstm_fused_cell/BlockLSTM:1lstm_fused_cell/concat/axis*
T0*
N*4
_output_shapes"
 :������������������d
j
 lstm_fused_cell/ExpandDims_2/dimConst*
valueB: *
dtype0*
_output_shapes
:
�
lstm_fused_cell/ExpandDims_2
ExpandDimslstm_fused_cell/zeros lstm_fused_cell/ExpandDims_2/dim*
T0*+
_output_shapes
:���������d
_
lstm_fused_cell/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lstm_fused_cell/concat_1ConcatV2lstm_fused_cell/ExpandDims_2lstm_fused_cell/mullstm_fused_cell/concat_1/axis*
T0*
N*4
_output_shapes"
 :������������������d
]
lstm_fused_cell/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
]
lstm_fused_cell/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
lstm_fused_cell/rangeRangelstm_fused_cell/range/startlstm_fused_cell/strided_slicelstm_fused_cell/range/delta*#
_output_shapes
:���������
�
lstm_fused_cell/stack_1Packnwordslstm_fused_cell/range*
T0*

axis*
N*'
_output_shapes
:���������
�
lstm_fused_cell/GatherNdGatherNdlstm_fused_cell/concatlstm_fused_cell/stack_1*
Tindices0*
Tparams0*'
_output_shapes
:���������d
_
lstm_fused_cell/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
lstm_fused_cell/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
lstm_fused_cell/range_1Rangelstm_fused_cell/range_1/startlstm_fused_cell/strided_slicelstm_fused_cell/range_1/delta*#
_output_shapes
:���������
�
lstm_fused_cell/stack_2Packnwordslstm_fused_cell/range_1*
T0*

axis*
N*'
_output_shapes
:���������
�
lstm_fused_cell/GatherNd_1GatherNdlstm_fused_cell/concat_1lstm_fused_cell/stack_2*
Tindices0*
Tparams0*'
_output_shapes
:���������d
�
ReverseSequenceReverseSequence	transposenwords*
	batch_dim*
T0*
seq_dim *5
_output_shapes#
!:�������������������*

Tlen0
�
9lstm_fused_cell_1/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@lstm_fused_cell_1/kernel*
valueB"�  �  *
dtype0*
_output_shapes
:
�
7lstm_fused_cell_1/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@lstm_fused_cell_1/kernel*
valueB
 *��*
dtype0*
_output_shapes
: 
�
7lstm_fused_cell_1/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@lstm_fused_cell_1/kernel*
valueB
 *�=*
dtype0*
_output_shapes
: 
�
Alstm_fused_cell_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9lstm_fused_cell_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@lstm_fused_cell_1/kernel*
dtype0* 
_output_shapes
:
��
�
7lstm_fused_cell_1/kernel/Initializer/random_uniform/subSub7lstm_fused_cell_1/kernel/Initializer/random_uniform/max7lstm_fused_cell_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@lstm_fused_cell_1/kernel*
_output_shapes
: 
�
7lstm_fused_cell_1/kernel/Initializer/random_uniform/mulMulAlstm_fused_cell_1/kernel/Initializer/random_uniform/RandomUniform7lstm_fused_cell_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@lstm_fused_cell_1/kernel* 
_output_shapes
:
��
�
3lstm_fused_cell_1/kernel/Initializer/random_uniformAdd7lstm_fused_cell_1/kernel/Initializer/random_uniform/mul7lstm_fused_cell_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@lstm_fused_cell_1/kernel* 
_output_shapes
:
��
�
lstm_fused_cell_1/kernel
VariableV2*
shape:
��*+
_class!
loc:@lstm_fused_cell_1/kernel*
dtype0* 
_output_shapes
:
��
�
lstm_fused_cell_1/kernel/AssignAssignlstm_fused_cell_1/kernel3lstm_fused_cell_1/kernel/Initializer/random_uniform*
T0*+
_class!
loc:@lstm_fused_cell_1/kernel* 
_output_shapes
:
��
�
lstm_fused_cell_1/kernel/readIdentitylstm_fused_cell_1/kernel*
T0*+
_class!
loc:@lstm_fused_cell_1/kernel* 
_output_shapes
:
��
�
(lstm_fused_cell_1/bias/Initializer/ConstConst*)
_class
loc:@lstm_fused_cell_1/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
lstm_fused_cell_1/bias
VariableV2*
shape:�*)
_class
loc:@lstm_fused_cell_1/bias*
dtype0*
_output_shapes	
:�
�
lstm_fused_cell_1/bias/AssignAssignlstm_fused_cell_1/bias(lstm_fused_cell_1/bias/Initializer/Const*
T0*)
_class
loc:@lstm_fused_cell_1/bias*
_output_shapes	
:�
�
lstm_fused_cell_1/bias/readIdentitylstm_fused_cell_1/bias*
T0*)
_class
loc:@lstm_fused_cell_1/bias*
_output_shapes	
:�
V
lstm_fused_cell_1/ShapeShapeReverseSequence*
T0*
_output_shapes
:
o
%lstm_fused_cell_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
q
'lstm_fused_cell_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'lstm_fused_cell_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
lstm_fused_cell_1/strided_sliceStridedSlicelstm_fused_cell_1/Shape%lstm_fused_cell_1/strided_slice/stack'lstm_fused_cell_1/strided_slice/stack_1'lstm_fused_cell_1/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
X
lstm_fused_cell_1/Shape_1ShapeReverseSequence*
T0*
_output_shapes
:
q
'lstm_fused_cell_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)lstm_fused_cell_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)lstm_fused_cell_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
!lstm_fused_cell_1/strided_slice_1StridedSlicelstm_fused_cell_1/Shape_1'lstm_fused_cell_1/strided_slice_1/stack)lstm_fused_cell_1/strided_slice_1/stack_1)lstm_fused_cell_1/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
[
lstm_fused_cell_1/stack/1Const*
value	B :d*
dtype0*
_output_shapes
: 
�
lstm_fused_cell_1/stackPacklstm_fused_cell_1/strided_slicelstm_fused_cell_1/stack/1*
T0*
N*
_output_shapes
:
b
lstm_fused_cell_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
lstm_fused_cell_1/zerosFilllstm_fused_cell_1/stacklstm_fused_cell_1/zeros/Const*
T0*'
_output_shapes
:���������d
X
lstm_fused_cell_1/Shape_2ShapeReverseSequence*
T0*
_output_shapes
:
q
'lstm_fused_cell_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)lstm_fused_cell_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)lstm_fused_cell_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
!lstm_fused_cell_1/strided_slice_2StridedSlicelstm_fused_cell_1/Shape_2'lstm_fused_cell_1/strided_slice_2/stack)lstm_fused_cell_1/strided_slice_2/stack_1)lstm_fused_cell_1/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
f
lstm_fused_cell_1/zeros_1Const*
valueBd*    *
dtype0*
_output_shapes
:d
a
lstm_fused_cell_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
^
lstm_fused_cell_1/MaxMaxnwordslstm_fused_cell_1/Const*
T0*
_output_shapes
: 
h
lstm_fused_cell_1/ToInt64Castlstm_fused_cell_1/Max*

SrcT0*
_output_shapes
: *

DstT0	
�
lstm_fused_cell_1/BlockLSTM	BlockLSTMlstm_fused_cell_1/ToInt64ReverseSequencelstm_fused_cell_1/zeroslstm_fused_cell_1/zeroslstm_fused_cell_1/kernel/readlstm_fused_cell_1/zeros_1lstm_fused_cell_1/zeros_1lstm_fused_cell_1/zeros_1lstm_fused_cell_1/bias/read*
	cell_clip%  ��*
T0*�
_output_shapes�
�:������������������d:������������������d:������������������d:������������������d:������������������d:������������������d:������������������d
f
$lstm_fused_cell_1/SequenceMask/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
h
&lstm_fused_cell_1/SequenceMask/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
$lstm_fused_cell_1/SequenceMask/RangeRange$lstm_fused_cell_1/SequenceMask/Const!lstm_fused_cell_1/strided_slice_1&lstm_fused_cell_1/SequenceMask/Const_1*#
_output_shapes
:���������
x
-lstm_fused_cell_1/SequenceMask/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
)lstm_fused_cell_1/SequenceMask/ExpandDims
ExpandDimsnwords-lstm_fused_cell_1/SequenceMask/ExpandDims/dim*
T0*'
_output_shapes
:���������
�
#lstm_fused_cell_1/SequenceMask/CastCast)lstm_fused_cell_1/SequenceMask/ExpandDims*

SrcT0*'
_output_shapes
:���������*

DstT0
�
#lstm_fused_cell_1/SequenceMask/LessLess$lstm_fused_cell_1/SequenceMask/Range#lstm_fused_cell_1/SequenceMask/Cast*
T0*0
_output_shapes
:������������������
�
%lstm_fused_cell_1/SequenceMask/Cast_1Cast#lstm_fused_cell_1/SequenceMask/Less*

SrcT0
*0
_output_shapes
:������������������*

DstT0
q
 lstm_fused_cell_1/transpose/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
lstm_fused_cell_1/transpose	Transpose%lstm_fused_cell_1/SequenceMask/Cast_1 lstm_fused_cell_1/transpose/perm*
T0*0
_output_shapes
:������������������
s
 lstm_fused_cell_1/ExpandDims/dimConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
lstm_fused_cell_1/ExpandDims
ExpandDimslstm_fused_cell_1/transpose lstm_fused_cell_1/ExpandDims/dim*
T0*4
_output_shapes"
 :������������������
u
 lstm_fused_cell_1/Tile/multiplesConst*!
valueB"      d   *
dtype0*
_output_shapes
:
�
lstm_fused_cell_1/TileTilelstm_fused_cell_1/ExpandDims lstm_fused_cell_1/Tile/multiples*
T0*4
_output_shapes"
 :������������������d
�
lstm_fused_cell_1/mulMullstm_fused_cell_1/BlockLSTM:6lstm_fused_cell_1/Tile*
T0*4
_output_shapes"
 :������������������d
l
"lstm_fused_cell_1/ExpandDims_1/dimConst*
valueB: *
dtype0*
_output_shapes
:
�
lstm_fused_cell_1/ExpandDims_1
ExpandDimslstm_fused_cell_1/zeros"lstm_fused_cell_1/ExpandDims_1/dim*
T0*+
_output_shapes
:���������d
_
lstm_fused_cell_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lstm_fused_cell_1/concatConcatV2lstm_fused_cell_1/ExpandDims_1lstm_fused_cell_1/BlockLSTM:1lstm_fused_cell_1/concat/axis*
T0*
N*4
_output_shapes"
 :������������������d
l
"lstm_fused_cell_1/ExpandDims_2/dimConst*
valueB: *
dtype0*
_output_shapes
:
�
lstm_fused_cell_1/ExpandDims_2
ExpandDimslstm_fused_cell_1/zeros"lstm_fused_cell_1/ExpandDims_2/dim*
T0*+
_output_shapes
:���������d
a
lstm_fused_cell_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lstm_fused_cell_1/concat_1ConcatV2lstm_fused_cell_1/ExpandDims_2lstm_fused_cell_1/mullstm_fused_cell_1/concat_1/axis*
T0*
N*4
_output_shapes"
 :������������������d
_
lstm_fused_cell_1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
lstm_fused_cell_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
lstm_fused_cell_1/rangeRangelstm_fused_cell_1/range/startlstm_fused_cell_1/strided_slicelstm_fused_cell_1/range/delta*#
_output_shapes
:���������
�
lstm_fused_cell_1/stack_1Packnwordslstm_fused_cell_1/range*
T0*

axis*
N*'
_output_shapes
:���������
�
lstm_fused_cell_1/GatherNdGatherNdlstm_fused_cell_1/concatlstm_fused_cell_1/stack_1*
Tindices0*
Tparams0*'
_output_shapes
:���������d
a
lstm_fused_cell_1/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
a
lstm_fused_cell_1/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
lstm_fused_cell_1/range_1Rangelstm_fused_cell_1/range_1/startlstm_fused_cell_1/strided_slicelstm_fused_cell_1/range_1/delta*#
_output_shapes
:���������
�
lstm_fused_cell_1/stack_2Packnwordslstm_fused_cell_1/range_1*
T0*

axis*
N*'
_output_shapes
:���������
�
lstm_fused_cell_1/GatherNd_1GatherNdlstm_fused_cell_1/concat_1lstm_fused_cell_1/stack_2*
Tindices0*
Tparams0*'
_output_shapes
:���������d
�
ReverseSequence_1ReverseSequencelstm_fused_cell_1/mulnwords*
	batch_dim*
T0*
seq_dim *4
_output_shapes"
 :������������������d*

Tlen0
X
concat_1/axisConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
concat_1ConcatV2lstm_fused_cell/mulReverseSequence_1concat_1/axis*
T0*
N*5
_output_shapes#
!:�������������������
e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
t
transpose_1	Transposeconcat_1transpose_1/perm*
T0*5
_output_shapes#
!:�������������������
k
dropout_2/IdentityIdentitytranspose_1*
T0*5
_output_shapes#
!:�������������������
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *��.�*
dtype0*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *��.>*
dtype0*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	�
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�

dense/kernel
VariableV2*
shape:	�*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	�
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
�
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB*    *
dtype0*
_output_shapes
:
q

dense/bias
VariableV2*
shape:*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:
�
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_class
loc:@dense/bias*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:
^
dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
e
dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
W
dense/Tensordot/ShapeShapedropout_2/Identity*
T0*
_output_shapes
:
_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shapedense/Tensordot/freedense/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
a
dense/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shapedense/Tensordot/axesdense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
n
dense/Tensordot/ProdProddense/Tensordot/GatherV2dense/Tensordot/Const*
T0*
_output_shapes
: 
a
dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
t
dense/Tensordot/Prod_1Proddense/Tensordot/GatherV2_1dense/Tensordot/Const_1*
T0*
_output_shapes
: 
]
dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
dense/Tensordot/concatConcatV2dense/Tensordot/freedense/Tensordot/axesdense/Tensordot/concat/axis*
T0*
N*
_output_shapes
:
y
dense/Tensordot/stackPackdense/Tensordot/Proddense/Tensordot/Prod_1*
T0*
N*
_output_shapes
:
�
dense/Tensordot/transpose	Transposedropout_2/Identitydense/Tensordot/concat*
T0*5
_output_shapes#
!:�������������������
�
dense/Tensordot/ReshapeReshapedense/Tensordot/transposedense/Tensordot/stack*
T0*0
_output_shapes
:������������������
q
 dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
dense/Tensordot/transpose_1	Transposedense/kernel/read dense/Tensordot/transpose_1/perm*
T0*
_output_shapes
:	�
p
dense/Tensordot/Reshape_1/shapeConst*
valueB"�      *
dtype0*
_output_shapes
:
�
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1dense/Tensordot/Reshape_1/shape*
T0*
_output_shapes
:	�
�
dense/Tensordot/MatMulMatMuldense/Tensordot/Reshapedense/Tensordot/Reshape_1*
T0*'
_output_shapes
:���������
a
dense/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
_
dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
dense/Tensordot/concat_1ConcatV2dense/Tensordot/GatherV2dense/Tensordot/Const_2dense/Tensordot/concat_1/axis*
T0*
N*
_output_shapes
:
�
dense/TensordotReshapedense/Tensordot/MatMuldense/Tensordot/concat_1*
T0*4
_output_shapes"
 :������������������
y
dense/BiasAddBiasAdddense/Tensordotdense/bias/read*
T0*4
_output_shapes"
 :������������������
�
$crf/Initializer/random_uniform/shapeConst*
_class

loc:@crf*
valueB"      *
dtype0*
_output_shapes
:

"crf/Initializer/random_uniform/minConst*
_class

loc:@crf*
valueB
 *�5�*
dtype0*
_output_shapes
: 

"crf/Initializer/random_uniform/maxConst*
_class

loc:@crf*
valueB
 *�5?*
dtype0*
_output_shapes
: 
�
,crf/Initializer/random_uniform/RandomUniformRandomUniform$crf/Initializer/random_uniform/shape*
T0*
_class

loc:@crf*
dtype0*
_output_shapes

:
�
"crf/Initializer/random_uniform/subSub"crf/Initializer/random_uniform/max"crf/Initializer/random_uniform/min*
T0*
_class

loc:@crf*
_output_shapes
: 
�
"crf/Initializer/random_uniform/mulMul,crf/Initializer/random_uniform/RandomUniform"crf/Initializer/random_uniform/sub*
T0*
_class

loc:@crf*
_output_shapes

:
�
crf/Initializer/random_uniformAdd"crf/Initializer/random_uniform/mul"crf/Initializer/random_uniform/min*
T0*
_class

loc:@crf*
_output_shapes

:
k
crf
VariableV2*
shape
:*
_class

loc:@crf*
dtype0*
_output_shapes

:
z

crf/AssignAssigncrfcrf/Initializer/random_uniform*
T0*
_class

loc:@crf*
_output_shapes

:
Z
crf/readIdentitycrf*
T0*
_class

loc:@crf*
_output_shapes

:
D
Shape_1Shapedense/BiasAdd*
T0*
_output_shapes
:
_
strided_slice_5/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_5/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_5StridedSliceShape_1strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
I
Equal/yConst*
value	B :*
dtype0*
_output_shapes
: 
I
EqualEqualstrided_slice_5Equal/y*
T0*
_output_shapes
: 
F
cond/SwitchSwitchEqualEqual*
T0
*
_output_shapes
: : 
I
cond/switch_tIdentitycond/Switch:1*
T0
*
_output_shapes
: 
G
cond/switch_fIdentitycond/Switch*
T0
*
_output_shapes
: 
@
cond/pred_idIdentityEqual*
T0
*
_output_shapes
: 
w
cond/SqueezeSqueezecond/Squeeze/Switch:1*
squeeze_dims
*
T0*'
_output_shapes
:���������
�
cond/Squeeze/SwitchSwitchdense/BiasAddcond/pred_id*
T0* 
_class
loc:@dense/BiasAdd*T
_output_shapesB
@:������������������:������������������
g
cond/ArgMax/dimensionConst^cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
h
cond/ArgMaxArgMaxcond/Squeezecond/ArgMax/dimension*
T0*#
_output_shapes
:���������
e
cond/ExpandDims/dimConst^cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
q
cond/ExpandDims
ExpandDimscond/ArgMaxcond/ExpandDims/dim*
T0	*'
_output_shapes
:���������
l
cond/Max/reduction_indicesConst^cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
g
cond/MaxMaxcond/Squeezecond/Max/reduction_indices*
T0*#
_output_shapes
:���������
c
	cond/CastCastcond/ExpandDims*

SrcT0	*'
_output_shapes
:���������*

DstT0
g
cond/ExpandDims_1/dimConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
}
cond/ExpandDims_1
ExpandDimscond/ExpandDims_1/Switchcond/ExpandDims_1/dim*
T0*"
_output_shapes
:
�
cond/ExpandDims_1/SwitchSwitchcrf/readcond/pred_id*
T0*
_class

loc:@crf*(
_output_shapes
::
u
cond/Slice/beginConst^cond/switch_f*!
valueB"            *
dtype0*
_output_shapes
:
t
cond/Slice/sizeConst^cond/switch_f*!
valueB"����   ����*
dtype0*
_output_shapes
:
�

cond/SliceSlicecond/Slice/Switchcond/Slice/begincond/Slice/size*
T0*
Index0*+
_output_shapes
:���������
�
cond/Slice/SwitchSwitchdense/BiasAddcond/pred_id*
T0* 
_class
loc:@dense/BiasAdd*T
_output_shapesB
@:������������������:������������������
n
cond/Squeeze_1Squeeze
cond/Slice*
squeeze_dims
*
T0*'
_output_shapes
:���������
w
cond/Slice_1/beginConst^cond/switch_f*!
valueB"           *
dtype0*
_output_shapes
:
v
cond/Slice_1/sizeConst^cond/switch_f*!
valueB"������������*
dtype0*
_output_shapes
:
�
cond/Slice_1Slicecond/Slice/Switchcond/Slice_1/begincond/Slice_1/size*
T0*
Index0*4
_output_shapes"
 :������������������
\

cond/ConstConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
\

cond/sub/yConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Z
cond/subSubcond/sub/Switch
cond/sub/y*
T0*#
_output_shapes
:���������
�
cond/sub/SwitchSwitchnwordscond/pred_id*
T0*
_class
loc:@nwords*2
_output_shapes 
:���������:���������
[
cond/MaximumMaximum
cond/Constcond/sub*
T0*#
_output_shapes
:���������
_
cond/rnn/RankConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
f
cond/rnn/range/startConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
f
cond/rnn/range/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
n
cond/rnn/rangeRangecond/rnn/range/startcond/rnn/Rankcond/rnn/range/delta*
_output_shapes
:
y
cond/rnn/concat/values_0Const^cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
f
cond/rnn/concat/axisConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn/concatConcatV2cond/rnn/concat/values_0cond/rnn/rangecond/rnn/concat/axis*
T0*
N*
_output_shapes
:
}
cond/rnn/transpose	Transposecond/Slice_1cond/rnn/concat*
T0*4
_output_shapes"
 :������������������
`
cond/rnn/sequence_lengthIdentitycond/Maximum*
T0*#
_output_shapes
:���������
P
cond/rnn/ShapeShapecond/rnn/transpose*
T0*
_output_shapes
:
v
cond/rnn/strided_slice/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
x
cond/rnn/strided_slice/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
x
cond/rnn/strided_slice/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn/strided_sliceStridedSlicecond/rnn/Shapecond/rnn/strided_slice/stackcond/rnn/strided_slice/stack_1cond/rnn/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
X
cond/rnn/Shape_1Shapecond/rnn/sequence_length*
T0*
_output_shapes
:
\
cond/rnn/stackPackcond/rnn/strided_slice*
T0*
N*
_output_shapes
:
^
cond/rnn/EqualEqualcond/rnn/Shape_1cond/rnn/stack*
T0*
_output_shapes
:
h
cond/rnn/ConstConst^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
K
cond/rnn/AllAllcond/rnn/Equalcond/rnn/Const*
_output_shapes
: 
�
cond/rnn/Assert/ConstConst^cond/switch_f*I
value@B> B8Expected shape for Tensor cond/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
x
cond/rnn/Assert/Const_1Const^cond/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
�
cond/rnn/Assert/Assert/data_0Const^cond/switch_f*I
value@B> B8Expected shape for Tensor cond/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
~
cond/rnn/Assert/Assert/data_2Const^cond/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
�
cond/rnn/Assert/AssertAssertcond/rnn/Allcond/rnn/Assert/Assert/data_0cond/rnn/stackcond/rnn/Assert/Assert/data_2cond/rnn/Shape_1*
T
2
�
cond/rnn/CheckSeqLenIdentitycond/rnn/sequence_length^cond/rnn/Assert/Assert*
T0*#
_output_shapes
:���������
R
cond/rnn/Shape_2Shapecond/rnn/transpose*
T0*
_output_shapes
:
x
cond/rnn/strided_slice_1/stackConst^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
z
 cond/rnn/strided_slice_1/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
z
 cond/rnn/strided_slice_1/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn/strided_slice_1StridedSlicecond/rnn/Shape_2cond/rnn/strided_slice_1/stack cond/rnn/strided_slice_1/stack_1 cond/rnn/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
R
cond/rnn/Shape_3Shapecond/rnn/transpose*
T0*
_output_shapes
:
x
cond/rnn/strided_slice_2/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
z
 cond/rnn/strided_slice_2/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
z
 cond/rnn/strided_slice_2/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn/strided_slice_2StridedSlicecond/rnn/Shape_3cond/rnn/strided_slice_2/stack cond/rnn/strided_slice_2/stack_1 cond/rnn/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
i
cond/rnn/ExpandDims/dimConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
y
cond/rnn/ExpandDims
ExpandDimscond/rnn/strided_slice_2cond/rnn/ExpandDims/dim*
T0*
_output_shapes
:
j
cond/rnn/Const_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
h
cond/rnn/concat_1/axisConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn/concat_1ConcatV2cond/rnn/ExpandDimscond/rnn/Const_1cond/rnn/concat_1/axis*
T0*
N*
_output_shapes
:
f
cond/rnn/zeros/ConstConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
q
cond/rnn/zerosFillcond/rnn/concat_1cond/rnn/zeros/Const*
T0*'
_output_shapes
:���������
j
cond/rnn/Const_2Const^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
\
cond/rnn/MinMincond/rnn/CheckSeqLencond/rnn/Const_2*
T0*
_output_shapes
: 
j
cond/rnn/Const_3Const^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
\
cond/rnn/MaxMaxcond/rnn/CheckSeqLencond/rnn/Const_3*
T0*
_output_shapes
: 
_
cond/rnn/timeConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn/TensorArrayTensorArrayV3cond/rnn/strided_slice_1*$
element_shape:���������*4
tensor_array_namecond/rnn/dynamic_rnn/output_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
�
cond/rnn/TensorArray_1TensorArrayV3cond/rnn/strided_slice_1*$
element_shape:���������*3
tensor_array_namecond/rnn/dynamic_rnn/input_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
c
!cond/rnn/TensorArrayUnstack/ShapeShapecond/rnn/transpose*
T0*
_output_shapes
:
�
/cond/rnn/TensorArrayUnstack/strided_slice/stackConst^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
�
1cond/rnn/TensorArrayUnstack/strided_slice/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
1cond/rnn/TensorArrayUnstack/strided_slice/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
)cond/rnn/TensorArrayUnstack/strided_sliceStridedSlice!cond/rnn/TensorArrayUnstack/Shape/cond/rnn/TensorArrayUnstack/strided_slice/stack1cond/rnn/TensorArrayUnstack/strided_slice/stack_11cond/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
y
'cond/rnn/TensorArrayUnstack/range/startConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
y
'cond/rnn/TensorArrayUnstack/range/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
�
!cond/rnn/TensorArrayUnstack/rangeRange'cond/rnn/TensorArrayUnstack/range/start)cond/rnn/TensorArrayUnstack/strided_slice'cond/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:���������
�
Ccond/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3cond/rnn/TensorArray_1!cond/rnn/TensorArrayUnstack/rangecond/rnn/transposecond/rnn/TensorArray_1:1*
T0*%
_class
loc:@cond/rnn/transpose*
_output_shapes
: 
d
cond/rnn/Maximum/xConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
^
cond/rnn/MaximumMaximumcond/rnn/Maximum/xcond/rnn/Max*
T0*
_output_shapes
: 
h
cond/rnn/MinimumMinimumcond/rnn/strided_slice_1cond/rnn/Maximum*
T0*
_output_shapes
: 
r
 cond/rnn/while/iteration_counterConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn/while/EnterEnter cond/rnn/while/iteration_counter*
T0*
parallel_iterations *
_output_shapes
: *,

frame_namecond/rnn/while/while_context
�
cond/rnn/while/Enter_1Entercond/rnn/time*
T0*
parallel_iterations *
_output_shapes
: *,

frame_namecond/rnn/while/while_context
�
cond/rnn/while/Enter_2Entercond/rnn/TensorArray:1*
T0*
parallel_iterations *
_output_shapes
: *,

frame_namecond/rnn/while/while_context
�
cond/rnn/while/Enter_3Entercond/Squeeze_1*
T0*
parallel_iterations *'
_output_shapes
:���������*,

frame_namecond/rnn/while/while_context
}
cond/rnn/while/MergeMergecond/rnn/while/Entercond/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
cond/rnn/while/Merge_1Mergecond/rnn/while/Enter_1cond/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
cond/rnn/while/Merge_2Mergecond/rnn/while/Enter_2cond/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
cond/rnn/while/Merge_3Mergecond/rnn/while/Enter_3cond/rnn/while/NextIteration_3*
T0*
N*)
_output_shapes
:���������: 
m
cond/rnn/while/LessLesscond/rnn/while/Mergecond/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
cond/rnn/while/Less/EnterEntercond/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecond/rnn/while/while_context
s
cond/rnn/while/Less_1Lesscond/rnn/while/Merge_1cond/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
cond/rnn/while/Less_1/EnterEntercond/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecond/rnn/while/while_context
k
cond/rnn/while/LogicalAnd
LogicalAndcond/rnn/while/Lesscond/rnn/while/Less_1*
_output_shapes
: 
V
cond/rnn/while/LoopCondLoopCondcond/rnn/while/LogicalAnd*
_output_shapes
: 
�
cond/rnn/while/SwitchSwitchcond/rnn/while/Mergecond/rnn/while/LoopCond*
T0*'
_class
loc:@cond/rnn/while/Merge*
_output_shapes
: : 
�
cond/rnn/while/Switch_1Switchcond/rnn/while/Merge_1cond/rnn/while/LoopCond*
T0*)
_class
loc:@cond/rnn/while/Merge_1*
_output_shapes
: : 
�
cond/rnn/while/Switch_2Switchcond/rnn/while/Merge_2cond/rnn/while/LoopCond*
T0*)
_class
loc:@cond/rnn/while/Merge_2*
_output_shapes
: : 
�
cond/rnn/while/Switch_3Switchcond/rnn/while/Merge_3cond/rnn/while/LoopCond*
T0*)
_class
loc:@cond/rnn/while/Merge_3*:
_output_shapes(
&:���������:���������
]
cond/rnn/while/IdentityIdentitycond/rnn/while/Switch:1*
T0*
_output_shapes
: 
a
cond/rnn/while/Identity_1Identitycond/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
a
cond/rnn/while/Identity_2Identitycond/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
r
cond/rnn/while/Identity_3Identitycond/rnn/while/Switch_3:1*
T0*'
_output_shapes
:���������
p
cond/rnn/while/add/yConst^cond/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
cond/rnn/while/addAddcond/rnn/while/Identitycond/rnn/while/add/y*
T0*
_output_shapes
: 
�
 cond/rnn/while/TensorArrayReadV3TensorArrayReadV3&cond/rnn/while/TensorArrayReadV3/Entercond/rnn/while/Identity_1(cond/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������
�
&cond/rnn/while/TensorArrayReadV3/EnterEntercond/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namecond/rnn/while/while_context
�
(cond/rnn/while/TensorArrayReadV3/Enter_1EnterCcond/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namecond/rnn/while/while_context
�
cond/rnn/while/GreaterEqualGreaterEqualcond/rnn/while/Identity_1!cond/rnn/while/GreaterEqual/Enter*
T0*#
_output_shapes
:���������
�
!cond/rnn/while/GreaterEqual/EnterEntercond/rnn/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *#
_output_shapes
:���������*,

frame_namecond/rnn/while/while_context
y
cond/rnn/while/ExpandDims/dimConst^cond/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
cond/rnn/while/ExpandDims
ExpandDimscond/rnn/while/Identity_3cond/rnn/while/ExpandDims/dim*
T0*+
_output_shapes
:���������
�
cond/rnn/while/add_1Addcond/rnn/while/ExpandDimscond/rnn/while/add_1/Enter*
T0*+
_output_shapes
:���������
�
cond/rnn/while/add_1/EnterEntercond/ExpandDims_1*
T0*
is_constant(*
parallel_iterations *"
_output_shapes
:*,

frame_namecond/rnn/while/while_context
�
$cond/rnn/while/Max/reduction_indicesConst^cond/rnn/while/Identity*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn/while/MaxMaxcond/rnn/while/add_1$cond/rnn/while/Max/reduction_indices*
T0*'
_output_shapes
:���������
�
cond/rnn/while/add_2Add cond/rnn/while/TensorArrayReadV3cond/rnn/while/Max*
T0*'
_output_shapes
:���������
{
cond/rnn/while/ArgMax/dimensionConst^cond/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
cond/rnn/while/ArgMaxArgMaxcond/rnn/while/add_1cond/rnn/while/ArgMax/dimension*
T0*'
_output_shapes
:���������
s
cond/rnn/while/CastCastcond/rnn/while/ArgMax*

SrcT0	*'
_output_shapes
:���������*

DstT0
�
cond/rnn/while/SelectSelectcond/rnn/while/GreaterEqualcond/rnn/while/Select/Entercond/rnn/while/Cast*
T0*&
_class
loc:@cond/rnn/while/Cast*'
_output_shapes
:���������
�
cond/rnn/while/Select/EnterEntercond/rnn/zeros*
T0*&
_class
loc:@cond/rnn/while/Cast*
parallel_iterations *
is_constant(*'
_output_shapes
:���������*,

frame_namecond/rnn/while/while_context
�
cond/rnn/while/Select_1Selectcond/rnn/while/GreaterEqualcond/rnn/while/Identity_3cond/rnn/while/add_2*
T0*'
_class
loc:@cond/rnn/while/add_2*'
_output_shapes
:���������
�
2cond/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV38cond/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entercond/rnn/while/Identity_1cond/rnn/while/Selectcond/rnn/while/Identity_2*
T0*&
_class
loc:@cond/rnn/while/Cast*
_output_shapes
: 
�
8cond/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntercond/rnn/TensorArray*
T0*&
_class
loc:@cond/rnn/while/Cast*
parallel_iterations *
is_constant(*
_output_shapes
:*,

frame_namecond/rnn/while/while_context
r
cond/rnn/while/add_3/yConst^cond/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
cond/rnn/while/add_3Addcond/rnn/while/Identity_1cond/rnn/while/add_3/y*
T0*
_output_shapes
: 
b
cond/rnn/while/NextIterationNextIterationcond/rnn/while/add*
T0*
_output_shapes
: 
f
cond/rnn/while/NextIteration_1NextIterationcond/rnn/while/add_3*
T0*
_output_shapes
: 
�
cond/rnn/while/NextIteration_2NextIteration2cond/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
z
cond/rnn/while/NextIteration_3NextIterationcond/rnn/while/Select_1*
T0*'
_output_shapes
:���������
S
cond/rnn/while/ExitExitcond/rnn/while/Switch*
T0*
_output_shapes
: 
W
cond/rnn/while/Exit_1Exitcond/rnn/while/Switch_1*
T0*
_output_shapes
: 
W
cond/rnn/while/Exit_2Exitcond/rnn/while/Switch_2*
T0*
_output_shapes
: 
h
cond/rnn/while/Exit_3Exitcond/rnn/while/Switch_3*
T0*'
_output_shapes
:���������
�
+cond/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3cond/rnn/TensorArraycond/rnn/while/Exit_2*'
_class
loc:@cond/rnn/TensorArray*
_output_shapes
: 
�
%cond/rnn/TensorArrayStack/range/startConst^cond/switch_f*'
_class
loc:@cond/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
%cond/rnn/TensorArrayStack/range/deltaConst^cond/switch_f*'
_class
loc:@cond/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
cond/rnn/TensorArrayStack/rangeRange%cond/rnn/TensorArrayStack/range/start+cond/rnn/TensorArrayStack/TensorArraySizeV3%cond/rnn/TensorArrayStack/range/delta*'
_class
loc:@cond/rnn/TensorArray*#
_output_shapes
:���������
�
-cond/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3cond/rnn/TensorArraycond/rnn/TensorArrayStack/rangecond/rnn/while/Exit_2*$
element_shape:���������*'
_class
loc:@cond/rnn/TensorArray*
dtype0*4
_output_shapes"
 :������������������
j
cond/rnn/Const_4Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
a
cond/rnn/Rank_1Const^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
h
cond/rnn/range_1/startConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
h
cond/rnn/range_1/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
v
cond/rnn/range_1Rangecond/rnn/range_1/startcond/rnn/Rank_1cond/rnn/range_1/delta*
_output_shapes
:
{
cond/rnn/concat_2/values_0Const^cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
h
cond/rnn/concat_2/axisConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn/concat_2ConcatV2cond/rnn/concat_2/values_0cond/rnn/range_1cond/rnn/concat_2/axis*
T0*
N*
_output_shapes
:
�
cond/rnn/transpose_1	Transpose-cond/rnn/TensorArrayStack/TensorArrayGatherV3cond/rnn/concat_2*
T0*4
_output_shapes"
 :������������������
�
cond/ReverseSequenceReverseSequencecond/rnn/transpose_1cond/Maximum*
T0*
seq_dim*4
_output_shapes"
 :������������������*

Tlen0
i
cond/ArgMax_1/dimensionConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
u
cond/ArgMax_1ArgMaxcond/rnn/while/Exit_3cond/ArgMax_1/dimension*
T0*#
_output_shapes
:���������
_
cond/Cast_1Castcond/ArgMax_1*

SrcT0	*#
_output_shapes
:���������*

DstT0
p
cond/ExpandDims_2/dimConst^cond/switch_f*
valueB :
���������*
dtype0*
_output_shapes
: 
u
cond/ExpandDims_2
ExpandDimscond/Cast_1cond/ExpandDims_2/dim*
T0*'
_output_shapes
:���������
a
cond/rnn_1/RankConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
h
cond/rnn_1/range/startConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
h
cond/rnn_1/range/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
v
cond/rnn_1/rangeRangecond/rnn_1/range/startcond/rnn_1/Rankcond/rnn_1/range/delta*
_output_shapes
:
{
cond/rnn_1/concat/values_0Const^cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
h
cond/rnn_1/concat/axisConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn_1/concatConcatV2cond/rnn_1/concat/values_0cond/rnn_1/rangecond/rnn_1/concat/axis*
T0*
N*
_output_shapes
:
�
cond/rnn_1/transpose	Transposecond/ReverseSequencecond/rnn_1/concat*
T0*4
_output_shapes"
 :������������������
b
cond/rnn_1/sequence_lengthIdentitycond/Maximum*
T0*#
_output_shapes
:���������
T
cond/rnn_1/ShapeShapecond/rnn_1/transpose*
T0*
_output_shapes
:
x
cond/rnn_1/strided_slice/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
z
 cond/rnn_1/strided_slice/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
z
 cond/rnn_1/strided_slice/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn_1/strided_sliceStridedSlicecond/rnn_1/Shapecond/rnn_1/strided_slice/stack cond/rnn_1/strided_slice/stack_1 cond/rnn_1/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
\
cond/rnn_1/Shape_1Shapecond/rnn_1/sequence_length*
T0*
_output_shapes
:
`
cond/rnn_1/stackPackcond/rnn_1/strided_slice*
T0*
N*
_output_shapes
:
d
cond/rnn_1/EqualEqualcond/rnn_1/Shape_1cond/rnn_1/stack*
T0*
_output_shapes
:
j
cond/rnn_1/ConstConst^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
Q
cond/rnn_1/AllAllcond/rnn_1/Equalcond/rnn_1/Const*
_output_shapes
: 
�
cond/rnn_1/Assert/ConstConst^cond/switch_f*K
valueBB@ B:Expected shape for Tensor cond/rnn_1/sequence_length:0 is *
dtype0*
_output_shapes
: 
z
cond/rnn_1/Assert/Const_1Const^cond/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
�
cond/rnn_1/Assert/Assert/data_0Const^cond/switch_f*K
valueBB@ B:Expected shape for Tensor cond/rnn_1/sequence_length:0 is *
dtype0*
_output_shapes
: 
�
cond/rnn_1/Assert/Assert/data_2Const^cond/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
�
cond/rnn_1/Assert/AssertAssertcond/rnn_1/Allcond/rnn_1/Assert/Assert/data_0cond/rnn_1/stackcond/rnn_1/Assert/Assert/data_2cond/rnn_1/Shape_1*
T
2
�
cond/rnn_1/CheckSeqLenIdentitycond/rnn_1/sequence_length^cond/rnn_1/Assert/Assert*
T0*#
_output_shapes
:���������
V
cond/rnn_1/Shape_2Shapecond/rnn_1/transpose*
T0*
_output_shapes
:
z
 cond/rnn_1/strided_slice_1/stackConst^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
|
"cond/rnn_1/strided_slice_1/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
|
"cond/rnn_1/strided_slice_1/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn_1/strided_slice_1StridedSlicecond/rnn_1/Shape_2 cond/rnn_1/strided_slice_1/stack"cond/rnn_1/strided_slice_1/stack_1"cond/rnn_1/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
V
cond/rnn_1/Shape_3Shapecond/rnn_1/transpose*
T0*
_output_shapes
:
z
 cond/rnn_1/strided_slice_2/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
|
"cond/rnn_1/strided_slice_2/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
|
"cond/rnn_1/strided_slice_2/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn_1/strided_slice_2StridedSlicecond/rnn_1/Shape_3 cond/rnn_1/strided_slice_2/stack"cond/rnn_1/strided_slice_2/stack_1"cond/rnn_1/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
k
cond/rnn_1/ExpandDims/dimConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 

cond/rnn_1/ExpandDims
ExpandDimscond/rnn_1/strided_slice_2cond/rnn_1/ExpandDims/dim*
T0*
_output_shapes
:
l
cond/rnn_1/Const_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
j
cond/rnn_1/concat_1/axisConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn_1/concat_1ConcatV2cond/rnn_1/ExpandDimscond/rnn_1/Const_1cond/rnn_1/concat_1/axis*
T0*
N*
_output_shapes
:
h
cond/rnn_1/zeros/ConstConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
w
cond/rnn_1/zerosFillcond/rnn_1/concat_1cond/rnn_1/zeros/Const*
T0*'
_output_shapes
:���������
l
cond/rnn_1/Const_2Const^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
b
cond/rnn_1/MinMincond/rnn_1/CheckSeqLencond/rnn_1/Const_2*
T0*
_output_shapes
: 
l
cond/rnn_1/Const_3Const^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
b
cond/rnn_1/MaxMaxcond/rnn_1/CheckSeqLencond/rnn_1/Const_3*
T0*
_output_shapes
: 
a
cond/rnn_1/timeConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn_1/TensorArrayTensorArrayV3cond/rnn_1/strided_slice_1*$
element_shape:���������*6
tensor_array_name!cond/rnn_1/dynamic_rnn/output_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
�
cond/rnn_1/TensorArray_1TensorArrayV3cond/rnn_1/strided_slice_1*$
element_shape:���������*5
tensor_array_name cond/rnn_1/dynamic_rnn/input_0*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
g
#cond/rnn_1/TensorArrayUnstack/ShapeShapecond/rnn_1/transpose*
T0*
_output_shapes
:
�
1cond/rnn_1/TensorArrayUnstack/strided_slice/stackConst^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
�
3cond/rnn_1/TensorArrayUnstack/strided_slice/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
3cond/rnn_1/TensorArrayUnstack/strided_slice/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
�
+cond/rnn_1/TensorArrayUnstack/strided_sliceStridedSlice#cond/rnn_1/TensorArrayUnstack/Shape1cond/rnn_1/TensorArrayUnstack/strided_slice/stack3cond/rnn_1/TensorArrayUnstack/strided_slice/stack_13cond/rnn_1/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
{
)cond/rnn_1/TensorArrayUnstack/range/startConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
{
)cond/rnn_1/TensorArrayUnstack/range/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
�
#cond/rnn_1/TensorArrayUnstack/rangeRange)cond/rnn_1/TensorArrayUnstack/range/start+cond/rnn_1/TensorArrayUnstack/strided_slice)cond/rnn_1/TensorArrayUnstack/range/delta*#
_output_shapes
:���������
�
Econd/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3cond/rnn_1/TensorArray_1#cond/rnn_1/TensorArrayUnstack/rangecond/rnn_1/transposecond/rnn_1/TensorArray_1:1*
T0*'
_class
loc:@cond/rnn_1/transpose*
_output_shapes
: 
f
cond/rnn_1/Maximum/xConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
d
cond/rnn_1/MaximumMaximumcond/rnn_1/Maximum/xcond/rnn_1/Max*
T0*
_output_shapes
: 
n
cond/rnn_1/MinimumMinimumcond/rnn_1/strided_slice_1cond/rnn_1/Maximum*
T0*
_output_shapes
: 
t
"cond/rnn_1/while/iteration_counterConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn_1/while/EnterEnter"cond/rnn_1/while/iteration_counter*
T0*
parallel_iterations *
_output_shapes
: *.

frame_name cond/rnn_1/while/while_context
�
cond/rnn_1/while/Enter_1Entercond/rnn_1/time*
T0*
parallel_iterations *
_output_shapes
: *.

frame_name cond/rnn_1/while/while_context
�
cond/rnn_1/while/Enter_2Entercond/rnn_1/TensorArray:1*
T0*
parallel_iterations *
_output_shapes
: *.

frame_name cond/rnn_1/while/while_context
�
cond/rnn_1/while/Enter_3Entercond/ExpandDims_2*
T0*
parallel_iterations *'
_output_shapes
:���������*.

frame_name cond/rnn_1/while/while_context
�
cond/rnn_1/while/MergeMergecond/rnn_1/while/Entercond/rnn_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
cond/rnn_1/while/Merge_1Mergecond/rnn_1/while/Enter_1 cond/rnn_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
cond/rnn_1/while/Merge_2Mergecond/rnn_1/while/Enter_2 cond/rnn_1/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
cond/rnn_1/while/Merge_3Mergecond/rnn_1/while/Enter_3 cond/rnn_1/while/NextIteration_3*
T0*
N*)
_output_shapes
:���������: 
s
cond/rnn_1/while/LessLesscond/rnn_1/while/Mergecond/rnn_1/while/Less/Enter*
T0*
_output_shapes
: 
�
cond/rnn_1/while/Less/EnterEntercond/rnn_1/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond/rnn_1/while/while_context
y
cond/rnn_1/while/Less_1Lesscond/rnn_1/while/Merge_1cond/rnn_1/while/Less_1/Enter*
T0*
_output_shapes
: 
�
cond/rnn_1/while/Less_1/EnterEntercond/rnn_1/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond/rnn_1/while/while_context
q
cond/rnn_1/while/LogicalAnd
LogicalAndcond/rnn_1/while/Lesscond/rnn_1/while/Less_1*
_output_shapes
: 
Z
cond/rnn_1/while/LoopCondLoopCondcond/rnn_1/while/LogicalAnd*
_output_shapes
: 
�
cond/rnn_1/while/SwitchSwitchcond/rnn_1/while/Mergecond/rnn_1/while/LoopCond*
T0*)
_class
loc:@cond/rnn_1/while/Merge*
_output_shapes
: : 
�
cond/rnn_1/while/Switch_1Switchcond/rnn_1/while/Merge_1cond/rnn_1/while/LoopCond*
T0*+
_class!
loc:@cond/rnn_1/while/Merge_1*
_output_shapes
: : 
�
cond/rnn_1/while/Switch_2Switchcond/rnn_1/while/Merge_2cond/rnn_1/while/LoopCond*
T0*+
_class!
loc:@cond/rnn_1/while/Merge_2*
_output_shapes
: : 
�
cond/rnn_1/while/Switch_3Switchcond/rnn_1/while/Merge_3cond/rnn_1/while/LoopCond*
T0*+
_class!
loc:@cond/rnn_1/while/Merge_3*:
_output_shapes(
&:���������:���������
a
cond/rnn_1/while/IdentityIdentitycond/rnn_1/while/Switch:1*
T0*
_output_shapes
: 
e
cond/rnn_1/while/Identity_1Identitycond/rnn_1/while/Switch_1:1*
T0*
_output_shapes
: 
e
cond/rnn_1/while/Identity_2Identitycond/rnn_1/while/Switch_2:1*
T0*
_output_shapes
: 
v
cond/rnn_1/while/Identity_3Identitycond/rnn_1/while/Switch_3:1*
T0*'
_output_shapes
:���������
t
cond/rnn_1/while/add/yConst^cond/rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
cond/rnn_1/while/addAddcond/rnn_1/while/Identitycond/rnn_1/while/add/y*
T0*
_output_shapes
: 
�
"cond/rnn_1/while/TensorArrayReadV3TensorArrayReadV3(cond/rnn_1/while/TensorArrayReadV3/Entercond/rnn_1/while/Identity_1*cond/rnn_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:���������
�
(cond/rnn_1/while/TensorArrayReadV3/EnterEntercond/rnn_1/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond/rnn_1/while/while_context
�
*cond/rnn_1/while/TensorArrayReadV3/Enter_1EnterEcond/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond/rnn_1/while/while_context
�
cond/rnn_1/while/GreaterEqualGreaterEqualcond/rnn_1/while/Identity_1#cond/rnn_1/while/GreaterEqual/Enter*
T0*#
_output_shapes
:���������
�
#cond/rnn_1/while/GreaterEqual/EnterEntercond/rnn_1/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *#
_output_shapes
:���������*.

frame_name cond/rnn_1/while/while_context
�
cond/rnn_1/while/SqueezeSqueezecond/rnn_1/while/Identity_3*
squeeze_dims
*
T0*#
_output_shapes
:���������
h
cond/rnn_1/while/ShapeShape"cond/rnn_1/while/TensorArrayReadV3*
T0*
_output_shapes
:
�
$cond/rnn_1/while/strided_slice/stackConst^cond/rnn_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
�
&cond/rnn_1/while/strided_slice/stack_1Const^cond/rnn_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
�
&cond/rnn_1/while/strided_slice/stack_2Const^cond/rnn_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
�
cond/rnn_1/while/strided_sliceStridedSlicecond/rnn_1/while/Shape$cond/rnn_1/while/strided_slice/stack&cond/rnn_1/while/strided_slice/stack_1&cond/rnn_1/while/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
z
cond/rnn_1/while/range/startConst^cond/rnn_1/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
z
cond/rnn_1/while/range/deltaConst^cond/rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
cond/rnn_1/while/rangeRangecond/rnn_1/while/range/startcond/rnn_1/while/strided_slicecond/rnn_1/while/range/delta*#
_output_shapes
:���������
�
cond/rnn_1/while/stackPackcond/rnn_1/while/rangecond/rnn_1/while/Squeeze*
T0*

axis*
N*'
_output_shapes
:���������
�
cond/rnn_1/while/GatherNdGatherNd"cond/rnn_1/while/TensorArrayReadV3cond/rnn_1/while/stack*
Tindices0*
Tparams0*#
_output_shapes
:���������
�
cond/rnn_1/while/ExpandDims/dimConst^cond/rnn_1/while/Identity*
valueB :
���������*
dtype0*
_output_shapes
: 
�
cond/rnn_1/while/ExpandDims
ExpandDimscond/rnn_1/while/GatherNdcond/rnn_1/while/ExpandDims/dim*
T0*'
_output_shapes
:���������
�
cond/rnn_1/while/SelectSelectcond/rnn_1/while/GreaterEqualcond/rnn_1/while/Select/Entercond/rnn_1/while/ExpandDims*
T0*.
_class$
" loc:@cond/rnn_1/while/ExpandDims*'
_output_shapes
:���������
�
cond/rnn_1/while/Select/EnterEntercond/rnn_1/zeros*
T0*.
_class$
" loc:@cond/rnn_1/while/ExpandDims*
parallel_iterations *
is_constant(*'
_output_shapes
:���������*.

frame_name cond/rnn_1/while/while_context
�
cond/rnn_1/while/Select_1Selectcond/rnn_1/while/GreaterEqualcond/rnn_1/while/Identity_3cond/rnn_1/while/ExpandDims*
T0*.
_class$
" loc:@cond/rnn_1/while/ExpandDims*'
_output_shapes
:���������
�
4cond/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3:cond/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Entercond/rnn_1/while/Identity_1cond/rnn_1/while/Selectcond/rnn_1/while/Identity_2*
T0*.
_class$
" loc:@cond/rnn_1/while/ExpandDims*
_output_shapes
: 
�
:cond/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntercond/rnn_1/TensorArray*
T0*.
_class$
" loc:@cond/rnn_1/while/ExpandDims*
parallel_iterations *
is_constant(*
_output_shapes
:*.

frame_name cond/rnn_1/while/while_context
v
cond/rnn_1/while/add_1/yConst^cond/rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
cond/rnn_1/while/add_1Addcond/rnn_1/while/Identity_1cond/rnn_1/while/add_1/y*
T0*
_output_shapes
: 
f
cond/rnn_1/while/NextIterationNextIterationcond/rnn_1/while/add*
T0*
_output_shapes
: 
j
 cond/rnn_1/while/NextIteration_1NextIterationcond/rnn_1/while/add_1*
T0*
_output_shapes
: 
�
 cond/rnn_1/while/NextIteration_2NextIteration4cond/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
~
 cond/rnn_1/while/NextIteration_3NextIterationcond/rnn_1/while/Select_1*
T0*'
_output_shapes
:���������
W
cond/rnn_1/while/ExitExitcond/rnn_1/while/Switch*
T0*
_output_shapes
: 
[
cond/rnn_1/while/Exit_1Exitcond/rnn_1/while/Switch_1*
T0*
_output_shapes
: 
[
cond/rnn_1/while/Exit_2Exitcond/rnn_1/while/Switch_2*
T0*
_output_shapes
: 
l
cond/rnn_1/while/Exit_3Exitcond/rnn_1/while/Switch_3*
T0*'
_output_shapes
:���������
�
-cond/rnn_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3cond/rnn_1/TensorArraycond/rnn_1/while/Exit_2*)
_class
loc:@cond/rnn_1/TensorArray*
_output_shapes
: 
�
'cond/rnn_1/TensorArrayStack/range/startConst^cond/switch_f*)
_class
loc:@cond/rnn_1/TensorArray*
value	B : *
dtype0*
_output_shapes
: 
�
'cond/rnn_1/TensorArrayStack/range/deltaConst^cond/switch_f*)
_class
loc:@cond/rnn_1/TensorArray*
value	B :*
dtype0*
_output_shapes
: 
�
!cond/rnn_1/TensorArrayStack/rangeRange'cond/rnn_1/TensorArrayStack/range/start-cond/rnn_1/TensorArrayStack/TensorArraySizeV3'cond/rnn_1/TensorArrayStack/range/delta*)
_class
loc:@cond/rnn_1/TensorArray*#
_output_shapes
:���������
�
/cond/rnn_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3cond/rnn_1/TensorArray!cond/rnn_1/TensorArrayStack/rangecond/rnn_1/while/Exit_2*$
element_shape:���������*)
_class
loc:@cond/rnn_1/TensorArray*
dtype0*4
_output_shapes"
 :������������������
l
cond/rnn_1/Const_4Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
c
cond/rnn_1/Rank_1Const^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
j
cond/rnn_1/range_1/startConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
j
cond/rnn_1/range_1/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
~
cond/rnn_1/range_1Rangecond/rnn_1/range_1/startcond/rnn_1/Rank_1cond/rnn_1/range_1/delta*
_output_shapes
:
}
cond/rnn_1/concat_2/values_0Const^cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
j
cond/rnn_1/concat_2/axisConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
cond/rnn_1/concat_2ConcatV2cond/rnn_1/concat_2/values_0cond/rnn_1/range_1cond/rnn_1/concat_2/axis*
T0*
N*
_output_shapes
:
�
cond/rnn_1/transpose_1	Transpose/cond/rnn_1/TensorArrayStack/TensorArrayGatherV3cond/rnn_1/concat_2*
T0*4
_output_shapes"
 :������������������
�
cond/Squeeze_2Squeezecond/rnn_1/transpose_1*
squeeze_dims
*
T0*0
_output_shapes
:������������������
b
cond/concat/axisConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
�
cond/concatConcatV2cond/ExpandDims_2cond/Squeeze_2cond/concat/axis*
T0*
N*0
_output_shapes
:������������������
�
cond/ReverseSequence_1ReverseSequencecond/concatcond/sub/Switch*
T0*
seq_dim*0
_output_shapes
:������������������*

Tlen0
n
cond/Max_1/reduction_indicesConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
t

cond/Max_1Maxcond/rnn/while/Exit_3cond/Max_1/reduction_indices*
T0*#
_output_shapes
:���������
|

cond/MergeMergecond/ReverseSequence_1	cond/Cast*
T0*
N*2
_output_shapes 
:������������������: 
d
cond/Merge_1Merge
cond/Max_1cond/Max*
T0*
N*%
_output_shapes
:���������: 
Y
index_to_string/ConstConst*
valueB	 BUNK*
dtype0*
_output_shapes
: 
�
index_to_stringHashTableV2*7
shared_name(&hash_table_../dev/vocab.tags.txt_-1_-2*
value_dtype0*
	key_dtype0	*
_output_shapes
: 

)index_to_string/table_init/asset_filepathConst*&
valueB B../dev/vocab.tags.txt*
dtype0*
_output_shapes
: 
�
index_to_string/table_initInitializeTableFromTextFileV2index_to_string)index_to_string/table_init/asset_filepath*
	key_index���������*
value_index���������
e
ToInt64Cast
cond/Merge*

SrcT0*0
_output_shapes
:������������������*

DstT0	
�
index_to_string_LookupLookupTableFindV2index_to_stringToInt64index_to_string/Const*

Tout0*0
_output_shapes
:������������������*	
Tin0	

initNoOp
�
init_all_tablesNoOp^index_to_string/table_init&^string_to_index/hash_table/table_init(^string_to_index_1/hash_table/table_init

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_0f536414866d4a0ab759cd491eebde7d/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�BVariableBchars_embeddingsBconv1d/biasBconv1d/kernelBcrfB
dense/biasBdense/kernelBglobal_stepBlstm_fused_cell/biasBlstm_fused_cell/kernelBlstm_fused_cell_1/biasBlstm_fused_cell_1/kernel*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariablechars_embeddingsconv1d/biasconv1d/kernelcrf
dense/biasdense/kernelglobal_steplstm_fused_cell/biaslstm_fused_cell/kernellstm_fused_cell_1/biaslstm_fused_cell_1/kernel"/device:CPU:0*
dtypes
2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
�
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BVariableBchars_embeddingsBconv1d/biasBconv1d/kernelBcrfB
dense/biasBdense/kernelBglobal_stepBlstm_fused_cell/biasBlstm_fused_cell/kernelBlstm_fused_cell_1/biasBlstm_fused_cell_1/kernel*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*D
_output_shapes2
0::::::::::::
v
save/AssignAssignVariablesave/RestoreV2*
T0*
_class
loc:@Variable*
_output_shapes
:	B�
�
save/Assign_1Assignchars_embeddingssave/RestoreV2:1*
T0*#
_class
loc:@chars_embeddings*
_output_shapes

:$d
{
save/Assign_2Assignconv1d/biassave/RestoreV2:2*
T0*
_class
loc:@conv1d/bias*
_output_shapes
:2
�
save/Assign_3Assignconv1d/kernelsave/RestoreV2:3*
T0* 
_class
loc:@conv1d/kernel*"
_output_shapes
:d2
o
save/Assign_4Assigncrfsave/RestoreV2:4*
T0*
_class

loc:@crf*
_output_shapes

:
y
save/Assign_5Assign
dense/biassave/RestoreV2:5*
T0*
_class
loc:@dense/bias*
_output_shapes
:
�
save/Assign_6Assigndense/kernelsave/RestoreV2:6*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
w
save/Assign_7Assignglobal_stepsave/RestoreV2:7*
T0	*
_class
loc:@global_step*
_output_shapes
: 
�
save/Assign_8Assignlstm_fused_cell/biassave/RestoreV2:8*
T0*'
_class
loc:@lstm_fused_cell/bias*
_output_shapes	
:�
�
save/Assign_9Assignlstm_fused_cell/kernelsave/RestoreV2:9*
T0*)
_class
loc:@lstm_fused_cell/kernel* 
_output_shapes
:
��
�
save/Assign_10Assignlstm_fused_cell_1/biassave/RestoreV2:10*
T0*)
_class
loc:@lstm_fused_cell_1/bias*
_output_shapes	
:�
�
save/Assign_11Assignlstm_fused_cell_1/kernelsave/RestoreV2:11*
T0*+
_class!
loc:@lstm_fused_cell_1/kernel* 
_output_shapes
:
��
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"�	
trainable_variables�	�
w
chars_embeddings:0chars_embeddings/Assignchars_embeddings/read:02-chars_embeddings/Initializer/random_uniform:08
k
conv1d/kernel:0conv1d/kernel/Assignconv1d/kernel/read:02*conv1d/kernel/Initializer/random_uniform:08
Z
conv1d/bias:0conv1d/bias/Assignconv1d/bias/read:02conv1d/bias/Initializer/zeros:08
�
lstm_fused_cell/kernel:0lstm_fused_cell/kernel/Assignlstm_fused_cell/kernel/read:023lstm_fused_cell/kernel/Initializer/random_uniform:08
~
lstm_fused_cell/bias:0lstm_fused_cell/bias/Assignlstm_fused_cell/bias/read:02(lstm_fused_cell/bias/Initializer/Const:08
�
lstm_fused_cell_1/kernel:0lstm_fused_cell_1/kernel/Assignlstm_fused_cell_1/kernel/read:025lstm_fused_cell_1/kernel/Initializer/random_uniform:08
�
lstm_fused_cell_1/bias:0lstm_fused_cell_1/bias/Assignlstm_fused_cell_1/bias/read:02*lstm_fused_cell_1/bias/Initializer/Const:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
C
crf:0
crf/Assign
crf/read:02 crf/Initializer/random_uniform:08"�

	variables�
�

X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
w
chars_embeddings:0chars_embeddings/Assignchars_embeddings/read:02-chars_embeddings/Initializer/random_uniform:08
k
conv1d/kernel:0conv1d/kernel/Assignconv1d/kernel/read:02*conv1d/kernel/Initializer/random_uniform:08
Z
conv1d/bias:0conv1d/bias/Assignconv1d/bias/read:02conv1d/bias/Initializer/zeros:08
H

Variable:0Variable/AssignVariable/read:02Variable/initial_value:0
�
lstm_fused_cell/kernel:0lstm_fused_cell/kernel/Assignlstm_fused_cell/kernel/read:023lstm_fused_cell/kernel/Initializer/random_uniform:08
~
lstm_fused_cell/bias:0lstm_fused_cell/bias/Assignlstm_fused_cell/bias/read:02(lstm_fused_cell/bias/Initializer/Const:08
�
lstm_fused_cell_1/kernel:0lstm_fused_cell_1/kernel/Assignlstm_fused_cell_1/kernel/read:025lstm_fused_cell_1/kernel/Initializer/random_uniform:08
�
lstm_fused_cell_1/bias:0lstm_fused_cell_1/bias/Assignlstm_fused_cell_1/bias/read:02*lstm_fused_cell_1/bias/Initializer/Const:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
C
crf:0
crf/Assign
crf/read:02 crf/Initializer/random_uniform:08"�
table_initializern
l
%string_to_index/hash_table/table_init
'string_to_index_1/hash_table/table_init
index_to_string/table_init"�
asset_filepaths�
�
6string_to_index/hash_table/table_init/asset_filepath:0
8string_to_index_1/hash_table/table_init/asset_filepath:0
+index_to_string/table_init/asset_filepath:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"�^
cond_context�]�]
�
cond/cond_textcond/pred_id:0cond/switch_t:0 *�
cond/ArgMax/dimension:0
cond/ArgMax:0
cond/Cast:0
cond/ExpandDims/dim:0
cond/ExpandDims:0
cond/Max/reduction_indices:0

cond/Max:0
cond/Squeeze/Switch:1
cond/Squeeze:0
cond/pred_id:0
cond/switch_t:0
dense/BiasAdd:0 
cond/pred_id:0cond/pred_id:0(
dense/BiasAdd:0cond/Squeeze/Switch:1
�[
cond/cond_text_1cond/pred_id:0cond/switch_f:0*�+
cond/ArgMax_1/dimension:0
cond/ArgMax_1:0
cond/Cast_1:0
cond/Const:0
cond/ExpandDims_1/Switch:0
cond/ExpandDims_1/dim:0
cond/ExpandDims_1:0
cond/ExpandDims_2/dim:0
cond/ExpandDims_2:0
cond/Max_1/reduction_indices:0
cond/Max_1:0
cond/Maximum:0
cond/ReverseSequence:0
cond/ReverseSequence_1:0
cond/Slice/Switch:0
cond/Slice/begin:0
cond/Slice/size:0
cond/Slice:0
cond/Slice_1/begin:0
cond/Slice_1/size:0
cond/Slice_1:0
cond/Squeeze_1:0
cond/Squeeze_2:0
cond/concat/axis:0
cond/concat:0
cond/pred_id:0
cond/rnn/All:0
cond/rnn/Assert/Assert/data_0:0
cond/rnn/Assert/Assert/data_2:0
cond/rnn/Assert/Const:0
cond/rnn/Assert/Const_1:0
cond/rnn/CheckSeqLen:0
cond/rnn/Const:0
cond/rnn/Const_1:0
cond/rnn/Const_2:0
cond/rnn/Const_3:0
cond/rnn/Const_4:0
cond/rnn/Equal:0
cond/rnn/ExpandDims/dim:0
cond/rnn/ExpandDims:0
cond/rnn/Max:0
cond/rnn/Maximum/x:0
cond/rnn/Maximum:0
cond/rnn/Min:0
cond/rnn/Minimum:0
cond/rnn/Rank:0
cond/rnn/Rank_1:0
cond/rnn/Shape:0
cond/rnn/Shape_1:0
cond/rnn/Shape_2:0
cond/rnn/Shape_3:0
cond/rnn/TensorArray:0
cond/rnn/TensorArray:1
/cond/rnn/TensorArrayStack/TensorArrayGatherV3:0
-cond/rnn/TensorArrayStack/TensorArraySizeV3:0
'cond/rnn/TensorArrayStack/range/delta:0
'cond/rnn/TensorArrayStack/range/start:0
!cond/rnn/TensorArrayStack/range:0
#cond/rnn/TensorArrayUnstack/Shape:0
Econd/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
)cond/rnn/TensorArrayUnstack/range/delta:0
)cond/rnn/TensorArrayUnstack/range/start:0
#cond/rnn/TensorArrayUnstack/range:0
1cond/rnn/TensorArrayUnstack/strided_slice/stack:0
3cond/rnn/TensorArrayUnstack/strided_slice/stack_1:0
3cond/rnn/TensorArrayUnstack/strided_slice/stack_2:0
+cond/rnn/TensorArrayUnstack/strided_slice:0
cond/rnn/TensorArray_1:0
cond/rnn/TensorArray_1:1
cond/rnn/concat/axis:0
cond/rnn/concat/values_0:0
cond/rnn/concat:0
cond/rnn/concat_1/axis:0
cond/rnn/concat_1:0
cond/rnn/concat_2/axis:0
cond/rnn/concat_2/values_0:0
cond/rnn/concat_2:0
cond/rnn/range/delta:0
cond/rnn/range/start:0
cond/rnn/range:0
cond/rnn/range_1/delta:0
cond/rnn/range_1/start:0
cond/rnn/range_1:0
cond/rnn/sequence_length:0
cond/rnn/stack:0
cond/rnn/strided_slice/stack:0
 cond/rnn/strided_slice/stack_1:0
 cond/rnn/strided_slice/stack_2:0
cond/rnn/strided_slice:0
 cond/rnn/strided_slice_1/stack:0
"cond/rnn/strided_slice_1/stack_1:0
"cond/rnn/strided_slice_1/stack_2:0
cond/rnn/strided_slice_1:0
 cond/rnn/strided_slice_2/stack:0
"cond/rnn/strided_slice_2/stack_1:0
"cond/rnn/strided_slice_2/stack_2:0
cond/rnn/strided_slice_2:0
cond/rnn/time:0
cond/rnn/transpose:0
cond/rnn/transpose_1:0
cond/rnn/while/Exit:0
cond/rnn/while/Exit_1:0
cond/rnn/while/Exit_2:0
cond/rnn/while/Exit_3:0
"cond/rnn/while/iteration_counter:0
cond/rnn/zeros/Const:0
cond/rnn/zeros:0
cond/rnn_1/All:0
!cond/rnn_1/Assert/Assert/data_0:0
!cond/rnn_1/Assert/Assert/data_2:0
cond/rnn_1/Assert/Const:0
cond/rnn_1/Assert/Const_1:0
cond/rnn_1/CheckSeqLen:0
cond/rnn_1/Const:0
cond/rnn_1/Const_1:0
cond/rnn_1/Const_2:0
cond/rnn_1/Const_3:0
cond/rnn_1/Const_4:0
cond/rnn_1/Equal:0
cond/rnn_1/ExpandDims/dim:0
cond/rnn_1/ExpandDims:0
cond/rnn_1/Max:0
cond/rnn_1/Maximum/x:0
cond/rnn_1/Maximum:0
cond/rnn_1/Min:0
cond/rnn_1/Minimum:0
cond/rnn_1/Rank:0
cond/rnn_1/Rank_1:0
cond/rnn_1/Shape:0
cond/rnn_1/Shape_1:0
cond/rnn_1/Shape_2:0
cond/rnn_1/Shape_3:0
cond/rnn_1/TensorArray:0
cond/rnn_1/TensorArray:1
1cond/rnn_1/TensorArrayStack/TensorArrayGatherV3:0
/cond/rnn_1/TensorArrayStack/TensorArraySizeV3:0
)cond/rnn_1/TensorArrayStack/range/delta:0
)cond/rnn_1/TensorArrayStack/range/start:0
#cond/rnn_1/TensorArrayStack/range:0
%cond/rnn_1/TensorArrayUnstack/Shape:0
Gcond/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
+cond/rnn_1/TensorArrayUnstack/range/delta:0
+cond/rnn_1/TensorArrayUnstack/range/start:0
%cond/rnn_1/TensorArrayUnstack/range:0
3cond/rnn_1/TensorArrayUnstack/strided_slice/stack:0
5cond/rnn_1/TensorArrayUnstack/strided_slice/stack_1:0
5cond/rnn_1/TensorArrayUnstack/strided_slice/stack_2:0
-cond/rnn_1/TensorArrayUnstack/strided_slice:0
cond/rnn_1/TensorArray_1:0
cond/rnn_1/TensorArray_1:1
cond/rnn_1/concat/axis:0
cond/rnn_1/concat/values_0:0
cond/rnn_1/concat:0
cond/rnn_1/concat_1/axis:0
cond/rnn_1/concat_1:0
cond/rnn_1/concat_2/axis:0
cond/rnn_1/concat_2/values_0:0
cond/rnn_1/concat_2:0
cond/rnn_1/range/delta:0
cond/rnn_1/range/start:0
cond/rnn_1/range:0
cond/rnn_1/range_1/delta:0
cond/rnn_1/range_1/start:0
cond/rnn_1/range_1:0
cond/rnn_1/sequence_length:0
cond/rnn_1/stack:0
 cond/rnn_1/strided_slice/stack:0
"cond/rnn_1/strided_slice/stack_1:0
"cond/rnn_1/strided_slice/stack_2:0
cond/rnn_1/strided_slice:0
"cond/rnn_1/strided_slice_1/stack:0
$cond/rnn_1/strided_slice_1/stack_1:0
$cond/rnn_1/strided_slice_1/stack_2:0
cond/rnn_1/strided_slice_1:0
"cond/rnn_1/strided_slice_2/stack:0
$cond/rnn_1/strided_slice_2/stack_1:0
$cond/rnn_1/strided_slice_2/stack_2:0
cond/rnn_1/strided_slice_2:0
cond/rnn_1/time:0
cond/rnn_1/transpose:0
cond/rnn_1/transpose_1:0
cond/rnn_1/while/Exit:0
cond/rnn_1/while/Exit_1:0
cond/rnn_1/while/Exit_2:0
cond/rnn_1/while/Exit_3:0
$cond/rnn_1/while/iteration_counter:0
cond/rnn_1/zeros/Const:0
cond/rnn_1/zeros:0
cond/sub/Switch:0
cond/sub/y:0

cond/sub:0
cond/switch_f:0

crf/read:0
dense/BiasAdd:0
nwords:0 
cond/pred_id:0cond/pred_id:0(

crf/read:0cond/ExpandDims_1/Switch:0
nwords:0cond/sub/Switch:0&
dense/BiasAdd:0cond/Slice/Switch:02��
cond/rnn/while/while_context *cond/rnn/while/LoopCond:02cond/rnn/while/Merge:0:cond/rnn/while/Identity:0Bcond/rnn/while/Exit:0Bcond/rnn/while/Exit_1:0Bcond/rnn/while/Exit_2:0Bcond/rnn/while/Exit_3:0J�
cond/ExpandDims_1:0
cond/rnn/CheckSeqLen:0
cond/rnn/Minimum:0
cond/rnn/TensorArray:0
Econd/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
cond/rnn/TensorArray_1:0
cond/rnn/strided_slice_1:0
!cond/rnn/while/ArgMax/dimension:0
cond/rnn/while/ArgMax:0
cond/rnn/while/Cast:0
cond/rnn/while/Enter:0
cond/rnn/while/Enter_1:0
cond/rnn/while/Enter_2:0
cond/rnn/while/Enter_3:0
cond/rnn/while/Exit:0
cond/rnn/while/Exit_1:0
cond/rnn/while/Exit_2:0
cond/rnn/while/Exit_3:0
cond/rnn/while/ExpandDims/dim:0
cond/rnn/while/ExpandDims:0
#cond/rnn/while/GreaterEqual/Enter:0
cond/rnn/while/GreaterEqual:0
cond/rnn/while/Identity:0
cond/rnn/while/Identity_1:0
cond/rnn/while/Identity_2:0
cond/rnn/while/Identity_3:0
cond/rnn/while/Less/Enter:0
cond/rnn/while/Less:0
cond/rnn/while/Less_1/Enter:0
cond/rnn/while/Less_1:0
cond/rnn/while/LogicalAnd:0
cond/rnn/while/LoopCond:0
&cond/rnn/while/Max/reduction_indices:0
cond/rnn/while/Max:0
cond/rnn/while/Merge:0
cond/rnn/while/Merge:1
cond/rnn/while/Merge_1:0
cond/rnn/while/Merge_1:1
cond/rnn/while/Merge_2:0
cond/rnn/while/Merge_2:1
cond/rnn/while/Merge_3:0
cond/rnn/while/Merge_3:1
cond/rnn/while/NextIteration:0
 cond/rnn/while/NextIteration_1:0
 cond/rnn/while/NextIteration_2:0
 cond/rnn/while/NextIteration_3:0
cond/rnn/while/Select/Enter:0
cond/rnn/while/Select:0
cond/rnn/while/Select_1:0
cond/rnn/while/Switch:0
cond/rnn/while/Switch:1
cond/rnn/while/Switch_1:0
cond/rnn/while/Switch_1:1
cond/rnn/while/Switch_2:0
cond/rnn/while/Switch_2:1
cond/rnn/while/Switch_3:0
cond/rnn/while/Switch_3:1
(cond/rnn/while/TensorArrayReadV3/Enter:0
*cond/rnn/while/TensorArrayReadV3/Enter_1:0
"cond/rnn/while/TensorArrayReadV3:0
:cond/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
4cond/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
cond/rnn/while/add/y:0
cond/rnn/while/add:0
cond/rnn/while/add_1/Enter:0
cond/rnn/while/add_1:0
cond/rnn/while/add_2:0
cond/rnn/while/add_3/y:0
cond/rnn/while/add_3:0
cond/rnn/zeros:09
cond/rnn/strided_slice_1:0cond/rnn/while/Less/Enter:0D
cond/rnn/TensorArray_1:0(cond/rnn/while/TensorArrayReadV3/Enter:0T
cond/rnn/TensorArray:0:cond/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0s
Econd/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0*cond/rnn/while/TensorArrayReadV3/Enter_1:03
cond/rnn/Minimum:0cond/rnn/while/Less_1/Enter:01
cond/rnn/zeros:0cond/rnn/while/Select/Enter:03
cond/ExpandDims_1:0cond/rnn/while/add_1/Enter:0=
cond/rnn/CheckSeqLen:0#cond/rnn/while/GreaterEqual/Enter:0Rcond/rnn/while/Enter:0Rcond/rnn/while/Enter_1:0Rcond/rnn/while/Enter_2:0Rcond/rnn/while/Enter_3:0Zcond/rnn/strided_slice_1:02��
cond/rnn_1/while/while_context *cond/rnn_1/while/LoopCond:02cond/rnn_1/while/Merge:0:cond/rnn_1/while/Identity:0Bcond/rnn_1/while/Exit:0Bcond/rnn_1/while/Exit_1:0Bcond/rnn_1/while/Exit_2:0Bcond/rnn_1/while/Exit_3:0J�
cond/rnn_1/CheckSeqLen:0
cond/rnn_1/Minimum:0
cond/rnn_1/TensorArray:0
Gcond/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
cond/rnn_1/TensorArray_1:0
cond/rnn_1/strided_slice_1:0
cond/rnn_1/while/Enter:0
cond/rnn_1/while/Enter_1:0
cond/rnn_1/while/Enter_2:0
cond/rnn_1/while/Enter_3:0
cond/rnn_1/while/Exit:0
cond/rnn_1/while/Exit_1:0
cond/rnn_1/while/Exit_2:0
cond/rnn_1/while/Exit_3:0
!cond/rnn_1/while/ExpandDims/dim:0
cond/rnn_1/while/ExpandDims:0
cond/rnn_1/while/GatherNd:0
%cond/rnn_1/while/GreaterEqual/Enter:0
cond/rnn_1/while/GreaterEqual:0
cond/rnn_1/while/Identity:0
cond/rnn_1/while/Identity_1:0
cond/rnn_1/while/Identity_2:0
cond/rnn_1/while/Identity_3:0
cond/rnn_1/while/Less/Enter:0
cond/rnn_1/while/Less:0
cond/rnn_1/while/Less_1/Enter:0
cond/rnn_1/while/Less_1:0
cond/rnn_1/while/LogicalAnd:0
cond/rnn_1/while/LoopCond:0
cond/rnn_1/while/Merge:0
cond/rnn_1/while/Merge:1
cond/rnn_1/while/Merge_1:0
cond/rnn_1/while/Merge_1:1
cond/rnn_1/while/Merge_2:0
cond/rnn_1/while/Merge_2:1
cond/rnn_1/while/Merge_3:0
cond/rnn_1/while/Merge_3:1
 cond/rnn_1/while/NextIteration:0
"cond/rnn_1/while/NextIteration_1:0
"cond/rnn_1/while/NextIteration_2:0
"cond/rnn_1/while/NextIteration_3:0
cond/rnn_1/while/Select/Enter:0
cond/rnn_1/while/Select:0
cond/rnn_1/while/Select_1:0
cond/rnn_1/while/Shape:0
cond/rnn_1/while/Squeeze:0
cond/rnn_1/while/Switch:0
cond/rnn_1/while/Switch:1
cond/rnn_1/while/Switch_1:0
cond/rnn_1/while/Switch_1:1
cond/rnn_1/while/Switch_2:0
cond/rnn_1/while/Switch_2:1
cond/rnn_1/while/Switch_3:0
cond/rnn_1/while/Switch_3:1
*cond/rnn_1/while/TensorArrayReadV3/Enter:0
,cond/rnn_1/while/TensorArrayReadV3/Enter_1:0
$cond/rnn_1/while/TensorArrayReadV3:0
<cond/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
6cond/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3:0
cond/rnn_1/while/add/y:0
cond/rnn_1/while/add:0
cond/rnn_1/while/add_1/y:0
cond/rnn_1/while/add_1:0
cond/rnn_1/while/range/delta:0
cond/rnn_1/while/range/start:0
cond/rnn_1/while/range:0
cond/rnn_1/while/stack:0
&cond/rnn_1/while/strided_slice/stack:0
(cond/rnn_1/while/strided_slice/stack_1:0
(cond/rnn_1/while/strided_slice/stack_2:0
 cond/rnn_1/while/strided_slice:0
cond/rnn_1/zeros:0A
cond/rnn_1/CheckSeqLen:0%cond/rnn_1/while/GreaterEqual/Enter:0=
cond/rnn_1/strided_slice_1:0cond/rnn_1/while/Less/Enter:0H
cond/rnn_1/TensorArray_1:0*cond/rnn_1/while/TensorArrayReadV3/Enter:0X
cond/rnn_1/TensorArray:0<cond/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0w
Gcond/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0,cond/rnn_1/while/TensorArrayReadV3/Enter_1:07
cond/rnn_1/Minimum:0cond/rnn_1/while/Less_1/Enter:05
cond/rnn_1/zeros:0cond/rnn_1/while/Select/Enter:0Rcond/rnn_1/while/Enter:0Rcond/rnn_1/while/Enter_1:0Rcond/rnn_1/while/Enter_2:0Rcond/rnn_1/while/Enter_3:0Zcond/rnn_1/strided_slice_1:0"�
saved_model_assets�*�
z
+type.googleapis.com/tensorflow.AssetFileDefK
8
6string_to_index/hash_table/table_init/asset_filepath:0vocab.words.txt
|
+type.googleapis.com/tensorflow.AssetFileDefM
:
8string_to_index_1/hash_table/table_init/asset_filepath:0vocab.chars.txt
n
+type.googleapis.com/tensorflow.AssetFileDef?
-
+index_to_string/table_init/asset_filepath:0vocab.tags.txt"%
saved_model_main_op


group_deps*�
serving_default�
=
chars4
chars:0'���������������������������
%
nwords
nwords:0���������
0
words'
words:0������������������
2
nchars(
nchars:0������������������8
pred_ids,
cond/Merge:0������������������@
tags8
index_to_string_Lookup:0������������������tensorflow/serving/predict