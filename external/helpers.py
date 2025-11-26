from inspect import trace
from operator import matmul
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import teaserpp_python

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1

def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT # PMC_EXACT / NONE
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS # FGR OR GNC-TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def normalizePointClouds(P_xyz, Q_xyz):
    """
    P and Q are two 3 by N matrices of points
    """
    min_A_cor = np.min(P_xyz, axis=1)
    max_A_cor = np.max(P_xyz, axis=1)
    min_B_cor = np.min(Q_xyz, axis=1)
    max_B_cor = np.max(Q_xyz, axis=1)
    scal_A = np.max(max_A_cor - min_A_cor)
    scal_B = np.max(max_B_cor - min_B_cor)
    scal = np.max([scal_A, scal_B])
    P_xyz = P_xyz / scal
    Q_xyz = Q_xyz / scal
    return P_xyz, Q_xyz, scal

def getBinaryTheta(theta):
    N = theta.shape[0]
    output = -1.0 * np.ones(N)
    output[theta] = 1.0
    return output

def get_teaser_certifier(noise_bound):
  certifier_params = teaserpp_python.DRSCertifier.Params()
  certifier_params.cbar2 = 1.0
  certifier_params.noise_bound = 2*noise_bound # need to double because this is the noise bound for TIMs
  certifier_params.sub_optimality = 1e-3
  certifier_params.max_iterations = 1e2
  certifier_params.gamma_tau = 1.8
  certifier = teaserpp_python.DRSCertifier(certifier_params)
  return certifier

def compute_transform_error(mat1, mat2):
  mat_err = matmul(mat1, np.linalg.inv(mat2))
  mat_err_R = mat_err[:3, :3]
  mat_err_t = mat_err[:3, 3]

  trace_R = np.trace(mat_err_R)
  if trace_R < -1:
    trace_R = -1

  err_t = np.linalg.norm(mat_err_t)
  err_R = np.arccos((trace_R - 1) / 2)

  return err_t, err_R