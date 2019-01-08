import cv2, ctypes, logging, os, numpy as np, pickle
import pyassimp as assimp, json
from collections import OrderedDict
from skimage.morphology import binary_closing, disk
import scipy


def sample_points_on_faces(vs, fs, rng, n_samples_per_face):
    idx = np.repeat(np.arange(fs.shape[0]), n_samples_per_face)

    r = rng.rand(idx.size, 2)
    r1 = r[:, :1]
    r2 = r[:, 1:]
    sqrt_r1 = np.sqrt(r1)

    v1 = vs[fs[idx, 0], :]
    v2 = vs[fs[idx, 1], :]
    v3 = vs[fs[idx, 2], :]
    pts = (1 - sqrt_r1) * v1 + sqrt_r1 * (1 - r2) * v2 + sqrt_r1 * r2 * v3

    v1 = vs[fs[:, 0], :]
    v2 = vs[fs[:, 1], :]
    v3 = vs[fs[:, 2], :]
    ar = 0.5 * np.sqrt(np.sum(np.cross(v1 - v3, v2 - v3) ** 2, 1))
    assert (np.all(ar >= 0))
    # ar[ar == 0] = 1
    # if not np.all(ar > 0):
    #   import pdb; pdb.set_trace()

    return pts, ar, idx


class Foo(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        str_ = ''
        for v in vars(self).keys():
            a = getattr(self, v)
            if True:  # isinstance(v, object):
                str__ = str(a)
                str__ = str__.replace('\n', '\n  ')
            else:
                str__ = str(a)
            str_ += '{:s}: {:s}'.format(v, str__)
            str_ += '\n'
        return str_


class Shape():
    def get_pyassimp_load_options(self):
        load_flags = assimp.postprocess.aiProcess_Triangulate
        load_flags = load_flags | assimp.postprocess.aiProcess_SortByPType
        load_flags = load_flags | assimp.postprocess.aiProcess_OptimizeGraph
        load_flags = load_flags | assimp.postprocess.aiProcess_OptimizeMeshes
        load_flags = load_flags | assimp.postprocess.aiProcess_RemoveRedundantMaterials
        load_flags = load_flags | assimp.postprocess.aiProcess_FindDegenerates
        load_flags = load_flags | assimp.postprocess.aiProcess_GenSmoothNormals
        load_flags = load_flags | assimp.postprocess.aiProcess_JoinIdenticalVertices
        load_flags = load_flags | assimp.postprocess.aiProcess_ImproveCacheLocality
        load_flags = load_flags | assimp.postprocess.aiProcess_GenUVCoords
        load_flags = load_flags | assimp.postprocess.aiProcess_FindInvalidData
        return load_flags

    def load_materials(self, meshes, dir_name, materials_scale):
        materials = []
        kk = ['opacity', 'transparent', 'refracti', 'ambient', 'diffuse', 'name',
              'specular', 'file', 'emissive', 'shadingm', 'shininess']
        for m in meshes:
            mat = {}
            for k in kk:
                if k == 'file':
                    mat[k] = None
                    if (k, 1) in m.material.properties:
                        mat[k] = m.material.properties[(k, 1)]
                else:
                    mat[k] = m.material.properties[(k, 0)]
            materials.append(mat)

        for _, m in zip(meshes, materials):
            m['file_img'] = None
            # print(m['file'] is not None, _.texturecoords.shape)
            if m['file'] is not None:
                # assert(_.texturecoords.shape[2] == 3)
                file_name = os.path.join(dir_name, m['file'])
                assert (os.path.exists(file_name)), \
                    'Texture file {:s} foes not exist.'.format(file_name)
                img_rgb = cv2.imread(file_name)[::-1, :, ::-1]
                if img_rgb.shape[0] != img_rgb.shape[1]:
                    logging.warn('Texture image not square.')
                    sz = np.maximum(img_rgb.shape[0], img_rgb.shape[1])
                    sz = int(np.power(2., np.ceil(np.log2(sz))))
                    sz = int(sz * materials_scale)
                    img_rgb = cv2.resize(img_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
                else:
                    sz = img_rgb.shape[0]
                    sz_ = int(np.power(2., np.ceil(np.log2(sz))))
                    if sz != sz_ or materials_scale != 1.:
                        logging.warn('Texture image not square of power of 2 size. ' +
                                     'Changing size from %d to %d.', sz, sz_)
                        sz = int(sz_ * materials_scale)
                        img_rgb = cv2.resize(img_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
                m['file_img'] = img_rgb
            if _.texturecoords.size == 0:
                _.texturecoords = np.zeros((1, _.vertices.shape[0], 3), dtype=np.float32)
        return materials

    def __init__(self, obj_file, material_file=None, load_materials=True,
                 name_prefix='', name_suffix='', materials_scale=1.0):
        if material_file is not None:
            logging.error('Ignoring material file input, reading them off obj file.')
        load_flags = self.get_pyassimp_load_options()
        scene = assimp.load(obj_file, processing=load_flags)
        filter_ind = self._filter_triangles(scene.meshes)
        self.meshes = [scene.meshes[i] for i in filter_ind]
        for i, m in enumerate(self.meshes):
            m.name = name_prefix + m.name + '_{:05d}'.format(i) + name_suffix
        logging.info('#Meshes: %d', len(self.meshes))

        dir_name = os.path.dirname(obj_file)
        # Load materials
        materials = [None for _ in self.meshes]
        if load_materials:
            materials = self.load_materials(self.meshes, dir_name, materials_scale)
        self.scene = scene
        self.materials = materials

    def _filter_triangles(self, meshes):
        select = []
        for i in range(len(meshes)):
            if meshes[i].primitivetypes == 4:
                select.append(i)
        return select

    def flip_shape(self):
        for m in self.meshes:
            m.vertices[:, 1] = -m.vertices[:, 1]
            m.normals[:, 1] = -m.normals[:, 1]
            bb = m.faces * 1
            bb[:, 1] = m.faces[:, 2]
            bb[:, 2] = m.faces[:, 1]
            m.faces = bb
            # m.vertices[:,[0,1]] = m.vertices[:,[1,0]]

    def make_z_up_suncg(self):
        for m in self.meshes:
            m.vertices = m.vertices[:, [2, 0, 1]]
            m.normals = m.normals[:, [2, 0, 1]]

    def get_vertices(self):
        vs = []
        for m in self.meshes:
            vs.append(m.vertices)
        vss = np.concatenate(vs, axis=0)
        return vss, vs

    def get_faces(self):
        vs = []
        for m in self.meshes:
            v = m.faces
            vs.append(v)
        return vs

    def get_number_of_meshes(self):
        return len(self.meshes)

    def scale(self, sx=1., sy=1., sz=1.):
        pass

    def sample_points_on_face_of_shape(self, i, n_samples_per_face, sc):
        v = self.meshes[i].vertices * sc
        f = self.meshes[i].faces
        p, face_areas, face_idx = sample_points_on_faces(
            v, f, np.random.RandomState(0), n_samples_per_face)
        return p, face_areas, face_idx

    def __del__(self):
        scene = self.scene
        assimp.release(scene)


def _project_to_map(map, vertex, wt=None, ignore_points_outside_map=False):
    """Projects points to map, returns how many points are present at each
    location."""
    num_points = np.zeros((map.size[1], map.size[0]))
    vertex_ = vertex[:, :2] - map.origin
    vertex_ = np.round(vertex_ / map.resolution).astype(np.int)
    if ignore_points_outside_map:
        good_ind = np.all(np.array([vertex_[:, 1] >= 0, vertex_[:, 1] < map.size[1],
                                    vertex_[:, 0] >= 0, vertex_[:, 0] < map.size[0]]),
                          axis=0)
        vertex_ = vertex_[good_ind, :]
        if wt is not None:
            wt = wt[good_ind]
    if wt is None:
        np.add.at(num_points, (vertex_[:, 1], vertex_[:, 0]), 1)
    else:
        assert (wt.shape[0] == vertex_.shape[0]), \
            'number of weights should be same as vertices.'
        np.add.at(num_points, (vertex_[:, 1], vertex_[:, 0]), wt)
    return num_points


def _get_xy_bounding_box(vertex, padding):
    """Returns the xy bounding box of the environment."""
    min_ = np.floor(np.min(vertex[:, :2], axis=0) - padding).astype(np.int)
    max_ = np.ceil(np.max(vertex[:, :2], axis=0) + padding).astype(np.int)
    return min_, max_


def make_map(padding, resolution, vertex, sc):
    """Returns a map structure."""
    min_, max_ = _get_xy_bounding_box(vertex * sc, padding=padding)
    sz = np.ceil((max_ - min_ + 1) / resolution).astype(np.int32)
    max_ = min_ + sz * resolution - 1
    map_struct = Foo(origin=min_, size=sz, max=max_, resolution=resolution,
                     padding=padding)
    return map_struct


def sample_and_project_points(shape, intervals, sc, map_struct, n_samples_per_face=200):
    num_points = np.zeros((map_struct.size[1], map_struct.size[0], len(intervals) - 1))

    for j in range(shape.get_number_of_meshes()):
        # p, face_areas, face_idx = shapes.sample_points_on_face_of_shape(
        #     j, n_samples_per_face, sc)
        # wt = face_areas[face_idx]/n_samples_per_face

        p, face_areas, face_idx = shape.sample_points_on_face_of_shape(
            j, 1, sc)
        n_samples_ = np.ceil((1. * face_areas) / map_struct.resolution / map_struct.resolution * n_samples_per_face)
        n_samples_ = np.ceil(np.mean(n_samples_)).astype(np.int32)
        p, face_areas, face_idx = shape.sample_points_on_face_of_shape(
            j, n_samples_, sc)
        wt = face_areas[face_idx] / n_samples_

        for k in range(len(intervals) - 1):
            lower, higher = intervals[k], intervals[k + 1]
            ind = np.all(np.concatenate((p[:, [2]] >= lower, p[:, [2]] < higher), axis=1), axis=1)
            num_points[:, :, k] += _project_to_map(map_struct, p[ind, :], wt[ind],
                                                   ignore_points_outside_map=True)

    map_struct.num_points = num_points
    return map_struct


def subplot(plt, Y_X, sz_y_sz_x=(10, 10), space_y_x=(0.1, 0.1), T=False):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams['figure.figsize'] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes_list


def make_map_json(house, resolution=0.05, sc=200.):
    min_ = house['levels'][0]['bbox']['min']
    min_ = np.array([min_[2], min_[0]])
    max_ = house['levels'][0]['bbox']['max']
    max_ = np.array([max_[2], max_[0]])
    min_ = np.floor(min_ / resolution) * resolution
    max_ = np.ceil(max_ / resolution) * resolution
    sz = np.floor((max_ - min_) / resolution).astype(int) + 1
    map_struct = Foo(origin=min_ * sc, size=sz, max=max_ * sc, resolution=resolution * sc)
    return map_struct


def pick_largest_cc(traversible):
    out = scipy.ndimage.label(traversible)[0]
    cnt = np.bincount(out.reshape(-1))[1:]
    return out == np.argmax(cnt) + 1

def build_collision_map(obj_file, json_file, save_file,
                        lower_limit, upper_limit,
                        resolution=0.05, N_SAMPLES_PER_FACE=200,
                        LARGEST_CC=True):
    with open(json_file) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    map_struct = make_map_json(data, resolution=resolution, sc=100.)

    tt = Shape(obj_file, load_materials=False)
    tt.make_z_up_suncg()
    # vss, vs = tt.get_vertices()
    # map_struct = make_map(10, resolution=5., vertex=vss, sc=100.)
    # print(map_struct)

    map_struct = sample_and_project_points(tt,
                                           [-np.inf, -10, lower_limit, upper_limit, np.inf],
                                           100., map_struct,
                                           N_SAMPLES_PER_FACE)
    yy = np.concatenate([map_struct.num_points[:, :, i] for i in range(map_struct.num_points.shape[2])], 0)
    yy = np.concatenate([yy, np.sum(map_struct.num_points, 2)], 0)

    total = np.sum(map_struct.num_points, 2) > 0
    traversible = np.all(np.array([total, map_struct.num_points[:, :, 2] == 0]), 0)
    if False:
        # Code for closing the small holes. Atleast a radius of 3 is way too big.
        traversible = np.invert(binary_closing(np.invert(traversible), disk(3)))
    if LARGEST_CC:
        traversible = pick_largest_cc(traversible)
    cv2.imwrite(save_file, traversible.astype(np.uint8) * 255)
    return traversible.astype(np.uint8)


def main():
    # cached_map_file = '../../../scratch/house/15517209609561f076b454da41022770/cachedmap0.05.pkl'
    # with open(cached_map_file, 'r') as f:
    #   mm = pickle.load(f)

    # root_dir = os.path.join('..', '..', '..', 'suncg_data', 'house')
    root_dir = './house'
    RESOLUTION = 0.05
    N_SAMPLES_PER_FACE = 200
    ROBOT_BASE = 10
    ROBOT_TOP = 120
    LARGEST_CC = True

    for house_name in ['4d7beaebb77aa8ab48bd81dd1801d447']:
        obj_file = os.path.join(root_dir, house_name, 'house.obj')
        json_file = os.path.join(root_dir, house_name, 'house.json')
        with open(json_file) as f:
            data = json.load(f, object_pairs_hook=OrderedDict)

        map_struct = make_map_json(data, resolution=RESOLUTION, sc=100.)

        tt = Shape(obj_file, load_materials=False)
        tt.make_z_up_suncg()
        # vss, vs = tt.get_vertices()
        # map_struct = make_map(10, resolution=5., vertex=vss, sc=100.)
        # print(map_struct)

        map_struct = sample_and_project_points(tt, [-np.inf, -10, ROBOT_BASE, ROBOT_TOP, np.inf], 100., map_struct,
                                               N_SAMPLES_PER_FACE)
        yy = np.concatenate([map_struct.num_points[:, :, i] for i in range(map_struct.num_points.shape[2])], 0)
        yy = np.concatenate([yy, np.sum(map_struct.num_points, 2)], 0)
        cv2.imwrite(house_name + '.png', (yy > 0).astype(np.uint8) * 255)

        total = np.sum(map_struct.num_points, 2) > 0
        traversible = np.all(np.array([total, map_struct.num_points[:, :, 2] == 0]), 0)
        if False:
            # Code for closing the small holes. Atleast a radius of 3 is way too big.
            traversible = np.invert(binary_closing(np.invert(traversible), disk(3)))
        if LARGEST_CC:
            traversible = pick_largest_cc(traversible)
        cv2.imwrite(house_name + '_valid.png', (total).astype(np.uint8) * 255)
        cv2.imwrite(house_name + '_walk.png', (traversible).astype(np.uint8) * 255)

        ref = cv2.imread(house_name + '_ref.png', cv2.IMREAD_UNCHANGED)
        cv2.imwrite(house_name + '_diff.png', (traversible).astype(np.uint8) * 100 + ref / 2)


if __name__ == '__main__':
    main()
