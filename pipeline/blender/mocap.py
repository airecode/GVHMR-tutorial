"""Retarget GVHMR (SMPL-X body) motion onto a rigkit armature (Blender, headless or live).

Frames: GVHMR global = gravity-aligned, Y up, SMPL faces +Z.  Blender: Z up, rig faces -Y.
C maps SMPL coords -> Blender (x, y, z) -> (x, -z, y).

Per bone b driven by SMPL joint j with SMPL rest direction (p -> c):
  A_b  = shortest arc  rig-rest-dir(b) -> SMPL-rest-dir(p->c)        (both in Blender world)
  D_b  = C R_global[j] C^T  @  A_b                                     (world delta of b)
so the posed rig bone points exactly along the posed SMPL bone and inherits SMPL's twist.
"""
import numpy as np
from mathutils import Matrix, Vector

C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)

# rig bone -> (SMPL joint providing the rotation, SMPL rest direction (from, to) or a fixed Blender vector)
MAP = {
    "Hips": (0, (0, 3)), "Spine": (3, (3, 6)), "Chest": (9, (9, 12)),
    "Neck": (12, (12, 15)), "Head": (15, (0, 0, 1)),
    "LShoulder": (13, (13, 16)), "LUpperArm": (16, (16, 18)), "LLowerArm": (18, (18, 20)), "LHand": (20, (18, 20)),
    "RShoulder": (14, (14, 17)), "RUpperArm": (17, (17, 19)), "RLowerArm": (19, (19, 21)), "RHand": (21, (19, 21)),
    "LUpperLeg": (1, (1, 4)), "LLowerLeg": (4, (4, 7)), "LFoot": (7, (7, 10)), "LToes": (10, (0, -1, 0)),
    "RUpperLeg": (2, (2, 5)), "RLowerLeg": (5, (5, 8)), "RFoot": (8, (8, 11)), "RToes": (11, (0, -1, 0)),
}


def rodrigues(aa):
    """(..., 3) axis-angle -> (..., 3, 3)"""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-8)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def fk(J, parents, orient, body_pose, transl):
    L = len(transl)
    local = rodrigues(np.concatenate([orient[:, None], body_pose.reshape(L, 21, 3)], 1))
    Rg = np.zeros_like(local)
    P = np.zeros((L, 22, 3))
    Rg[:, 0] = local[:, 0]
    P[:, 0] = J[0] + transl
    for j in range(1, 22):
        p = parents[j]
        Rg[:, j] = Rg[:, p] @ local[:, j]
        P[:, j] = P[:, p] + np.einsum("lij,j->li", Rg[:, p], J[j] - J[p])
    return Rg, P


def load(npz_path, smplx_path, contact_frames=24):
    d = np.load(npz_path)
    m = np.load(smplx_path, allow_pickle=True)
    betas = d["betas"].mean(0)
    v = m["v_template"] + m["shapedirs"][:, :, :len(betas)] @ betas
    J = (m["J_regressor"] @ v)[:22]
    parents = m["kintree_table"][0][:22].astype(int)
    L = len(d["transl"])
    Rg, P = fk(J, parents, d["global_orient"], d["body_pose"], d["transl"])
    # GVHMR's world-frame root height is pulled towards the ground by its motion prior (the flying kick
    # comes out ~0.5 m too low). The camera-frame result keeps the true image-space height, so fit the
    # floor plane through the lowest foot joint of the first (running) frames and measure the pelvis
    # height above that plane in camera space.
    _, Pc = fk(J, parents, d["incam_global_orient"], d["body_pose"], d["incam_transl"])
    feet = [7, 8, 10, 11]
    pts = np.array([Pc[f, feet[int(np.argmax(Pc[f, feet, 1]))]] for f in range(contact_frames)])
    c = pts.mean(0)
    n = np.linalg.svd(pts - c)[2][2]
    n = n if n[1] < 0 else -n  # camera y points down
    height = smooth((Pc[:, 0] - c) @ n, 3)  # pelvis above the contact level
    # to Blender frame
    Rb = C @ Rg @ C.T
    Jb = J @ C.T
    Pb = P @ C.T
    Rc, _ = fk(J, parents, d["incam_global_orient"], d["body_pose"], d["incam_transl"])
    # OpenCV camera -> Blender world, per frame (world root orientation vs camera-frame root orientation)
    cam_R = C @ Rg[:, 0] @ np.transpose(Rc[:, 0], (0, 2, 1))
    # video size: stored by run_pick_person.py; older files only have K (principal point = image centre)
    w = int(d["width"]) if "width" in d else int(round(2 * float(d["K"][0, 2])))
    h = int(d["height"]) if "height" in d else int(round(2 * float(d["K"][1, 2])))
    return dict(R=Rb, J=Jb, P=Pb, L=L, height=height, cam_R=cam_R, cam_pelvis=Pc[:, 0], K=d["K"],
                fps=float(d["fps"]) if "fps" in d else 24.0, size=(w, h))


def smooth(x, w=3):
    """centred moving average along axis 0 (edge padded)"""
    if w <= 1:
        return x
    pad = np.concatenate([np.repeat(x[:1], w // 2, 0), x, np.repeat(x[-1:], w // 2, 0)])
    k = np.ones(w) / w
    return np.apply_along_axis(lambda s: np.convolve(s, k, "valid"), 0, pad)


def retarget(arm, mc, action_name="mocap"):
    """Bake mocap onto arm. Returns the action."""
    bones = arm.data.bones
    B = {b.name: b.matrix_local.to_3x3().normalized() for b in bones}
    head = {b.name: b.head_local.copy() for b in bones}
    par = {b.name: (b.parent.name if b.parent else None) for b in bones}
    order = []
    todo = [b.name for b in bones]
    while todo:
        for n in list(todo):
            if par[n] is None or par[n] in order:
                order.append(n)
                todo.remove(n)
    J, R, P = mc["J"], mc["R"], mc["P"]

    # rest alignment per bone
    A = {}
    for n, (j, spec) in MAP.items():
        if n not in B:
            continue
        ur = (bones[n].tail_local - bones[n].head_local).normalized()
        us = Vector(spec) if len(spec) == 3 else Vector(J[spec[1]] - J[spec[0]])
        A[n] = ur.rotation_difference(us.normalized()).to_matrix()

    # root translation: scale by leg length, recentre, put the feet on the ground
    leg_rig = (head["LUpperLeg"] - head["LFoot"]).length + (head["RUpperLeg"] - head["RFoot"]).length
    leg_smpl = np.linalg.norm(J[1] - J[7]) + np.linalg.norm(J[2] - J[8])
    s = leg_rig / leg_smpl
    pelvis = P[:, 0] * s
    # vertical: camera-space height above the floor contact level (see load); the contact joints are the
    # SMPL ankle / foot joints, which sit about as high above the floor as the rig's toe joints
    contact = min(head["LToes"].z, head["RToes"].z, head["LFoot"].z, head["RFoot"].z)
    pelvis[:, 2] = mc["height"] * s + contact
    off = Vector((-pelvis[0, 0], -pelvis[0, 1], 0.0))
    hips_rest = head["Hips"]
    # where the SMPL pelvis sits in the rig's rest pose: between the rig's hip joints, plus SMPL's own
    # pelvis-above-hip-joints offset (scaled). The Hips bone keeps its rest offset from that point.
    pel_rest = (head["LUpperLeg"] + head["RUpperLeg"]) / 2 + Vector(J[0] - (J[1] + J[2]) / 2) * s
    print(f"[mocap] scale={s:.3f} pelvis z range {pelvis[:, 2].min():.2f}..{pelvis[:, 2].max():.2f}")

    mc["pelvis_rig"] = np.array([Vector(p) + off for p in pelvis])
    mc["scale"] = s
    arm.animation_data_create()
    import bpy
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    arm.animation_data.action = act
    prev = {}
    for f in range(mc["L"]):
        D = {}
        for n in order:
            if n in A:
                D[n] = Matrix(R[f, MAP[n][0]].tolist()) @ A[n]
            else:
                D[n] = D[par[n]] if par[n] else Matrix.Identity(3)
        for n in order:
            Dp = D[par[n]] if par[n] else Matrix.Identity(3)
            q = (B[n].transposed() @ Dp.transposed() @ D[n] @ B[n]).to_quaternion()
            if n in prev and prev[n].dot(q) < 0:
                q = -q
            prev[n] = q
            pb = arm.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=f + 1)
        # Hips location: posed SMPL pelvis (scaled) + the rig's rest pelvis->Hips offset, rotated with the hips
        pel = Vector(pelvis[f]) + off
        target = pel + Matrix(R[f, 0].tolist()) @ (hips_rest - pel_rest)
        pb = arm.pose.bones["Hips"]
        pb.location = B["Hips"].transposed() @ (target - hips_rest)
        pb.keyframe_insert("location", frame=f + 1)
    return act


def add_video_camera(mc, video_path, width=None, height=None, name="VideoCam", smooth_w=5):
    """Camera that reproduces the source video's view (GVHMR camera, anchored on the pelvis), with the video
    as its background so the retarget can be compared frame by frame."""
    import bpy
    from mathutils import Quaternion
    flip = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> Blender camera axes
    pos = mc["pelvis_rig"] - mc["scale"] * np.einsum("lij,lj->li", mc["cam_R"], mc["cam_pelvis"])
    pos = smooth(pos, smooth_w)
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"
    width, height = width or mc["size"][0], height or mc["size"][1]
    cd.sensor_width = 36.0
    cd.lens = float(mc["K"][0, 0]) / width * 36.0
    cd.clip_start, cd.clip_end = 0.05, 200
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    prev = None
    for f in range(mc["L"]):
        q = Matrix((mc["cam_R"][f] @ flip).tolist()).to_quaternion()
        if prev is not None and prev.dot(q) < 0:
            q = -q
        prev = q
        cam.location = Vector(pos[f])
        cam.rotation_quaternion = q
        cam.keyframe_insert("location", frame=f + 1)
        cam.keyframe_insert("rotation_quaternion", frame=f + 1)
    if video_path:
        clip = bpy.data.movieclips.load(video_path)
        cd.show_background_images = True
        bg = cd.background_images.new()
        bg.source = "MOVIE_CLIP"
        bg.clip = clip
        bg.alpha = 1.0
        bg.display_depth = "BACK"
        bg.frame_method = "FIT"
    return cam
