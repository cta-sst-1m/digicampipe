'''
testing digicam.utils.Camera here.
Not all features it inherits ... just the two new features are tested.

'''


def test_can_be_constructed_without_config_path():
    from digicampipe.utils import Camera
    cam = Camera()

    # This is calling the __init__ of its base-class. So I do not need to
    # test that that __init__ really worked.
    # just some spot checks maybe...

    assert len(cam.Pixels) == 1296
    assert len(cam.Modules) == 108
    assert len(cam.Clusters_7) == 432
    assert len(cam.Clusters_19) == 432


def has_attribute_geometry():
    from digicampipe.utils import Camera
    cam = Camera()

    g = cam.geometry

    # just some spot checks:
    assert g.neighbor_matrix.shape == (1296, 1296)
    assert str(g.pix_x.max()) == '425.25 mm'
