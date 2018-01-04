
def test_cone_simu():
    from digicampipe.image.cones_image import ConesImage, cones_simu
    from digicampipe.image.cones_image import simu_match

    # create an image  with a geometry compatible to the camera with
    # known angle, spacing between pixels etc...
    test_image, true_positions = cones_simu(
        offset=(-4.3, -2.1),
        angle_deg=10,
        pixel_radius=35,
    )
    cones_img = ConesImage(test_image)
    cones_img.get_cone(radius_mask=2.1, save_to_file=False)
    cones_img.fit_camera_geometry()
    cones_img.refine_camera_geometry()

    assert simu_match(cones_img, true_positions, std_error_max_px=0.5)
