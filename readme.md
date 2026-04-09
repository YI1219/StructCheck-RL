**Dataset Description**



This dataset is designed for linking BIM elements with multimodal data.



Each BIM element is associated with geometry, semantic attributes, visual observations, and temporal snapshots.



**1. Dataset Overview**



Each element is represented by four types of information:



BIM Semantics — static metadata and geometry



Temporal Semantics — multi-date observations



Local Images — projected visual regions



Point Cloud Summary — statistical descriptors



**2. Folder Structure**

DatasetRoot/

&nbsp;├── <Storey>/

&nbsp;│    ├── <IfcType>/

&nbsp;│    │    ├── <ElementID\_\_GUID>/

&nbsp;│    │    │    ├── meta.json                  # BIM attributes

&nbsp;│    │    │    ├── <run\_date>/                # Temporal Semantics

&nbsp;│    │    │    │    ├── bim_projection_contours.json       # Local image annotation

&nbsp;│    │    │    │    ├── .png           # Local image

&nbsp;│    │    │    │    ├── point_cloud_summary.json       # Point Cloud Summary



**3. Data Usage for Reinforcement Learning**

This work formulates construction progress assessment as a multimodal decision-making task. Each sample corresponds to a BIM element at a specific timestamp and integrates four types of semantic information.

1. Input Representation

Each training instance consists of the following four modalities:

BIM Semantics
Extracted from meta.json, including element name, IFC type, dimensions, and textual description. This modality defines the semantic category of the element.

Point Cloud Summary
Derived from point_cloud_summary.json, providing geometric evidence such as coverage, voxel overlap, and structural consistency between BIM and as-built data.

Temporal Context
Aggregated from historical progress_decision.json files across earlier timestamps. This modality captures the progression trend of the same element.

Local Image Observations
Obtained from annotated panoramic images and bim_projection_contours.json. The target region is explicitly defined by contour polygons, and the model is required to focus on these regions for visual reasoning.

2. Task Definition

Given the above multimodal inputs, the model is required to predict:

Stage_label:
The construction activity category inferred primarily from BIM semantics.

Completion:
The construction status, defined differently depending on element type:

Discrete elements: {Completed, Not Present, Uncertain}

Continuous elements: {Not Started, In Progress (low/medium/high), Completed, Uncertain}

Reason:
A structured explanation integrating visual, geometric, semantic, and temporal evidence.

3. Decision Principle

The model must prioritize evidence within the contour-defined regions in the local images. Information outside these regions should only be used as auxiliary context.

The final decision should be made by jointly considering all available modalities:

Visual presence within contour regions

Geometric consistency from point cloud data

Semantic definition from BIM attributes

Temporal progression from historical records


# =========================
# 读取 meta
# =========================

meta = read_json(META_PATH)

target_element_text = f"""
Name: "{safe(meta.get('Name'))}";
Description: "{safe(meta.get('说明'))}";
IfcType: "{safe(meta.get('IfcType'))}";
Height: "{safe(meta.get('Height'))}";
Width: "{safe(meta.get('Width'))}";
Length: "{safe(meta.get('Length'))}";
""".strip()

# ===== 基于这6个字段生成BIM attributes =====
bim_semantics = f"""
This element is identified as "{safe(meta.get('Name'))}", 
with IFC type "{safe(meta.get('IfcType'))}". 
It has approximate dimensions of height {safe(meta.get('Height'))}, 
width {safe(meta.get('Width'))}, and length {safe(meta.get('Length'))}. 
Additional description: "{safe(meta.get('说明'))}".
""".strip()


# =========================
# 读取 point cloud summary
# =========================
pc_summary = read_json(POINT_CLOUD_SUMMARY_PATH)

# ===== 基于这1个字段生成Point Cloud 的比较情况 =====
pointcloud_summary_text = f"""
SummaryLine: "{safe(pc_summary.get('summary_line'))}";
""".strip()

