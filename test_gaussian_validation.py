"""
Extended test for validation of centroid and RMS calculations
on analytically known Gaussian distribution.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from src.analysis.image_analyzer import ImageAnalyzer
import matplotlib.pyplot as plt


def create_2d_gaussian(size, sigma, center=None, amplitude=1.0):
    """
    Creates a 2D Gaussian distribution with known parameters.

    Args:
        size: Image size (size x size)
        sigma: Standard deviation (in pixels)
        center: Distribution center (x, y) in coordinates from center.
                Default (0, 0) - image center
        amplitude: Distribution amplitude

    Returns:
        2D array with Gaussian distribution
    """
    if center is None:
        center = (0, 0)

    # Create coordinate grid (centered, as in MATLAB)
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)

    # 2D Gaussian distribution
    # I(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
    gaussian = amplitude * np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))

    return gaussian


def test_gaussian_centered():
    """
    Test 1: Gaussian distribution centered at (0, 0).
    For an ideal Gaussian, the centroid should coincide with the center of distribution.
    RMS should be equal to sigma (standard deviation of the Gaussian).
    """
    print("\n" + "="*70)
    print("Test 1: Centered Gaussian distribution")
    print("="*70)

    analyzer = ImageAnalyzer()

    size = 101  # Odd size for clear center
    sigma_pixels = 10.0  # Standard deviation in pixels
    pixel_size = 0.0284  # mm (as in real camera)

    # Create ideal Gaussian centered at (0, 0)
    image = create_2d_gaussian(size, sigma_pixels, center=(0, 0))

    # Calculate centroid and RMS
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    # Expected values
    expected_centroid = 0.0  # At center
    expected_rms = sigma_pixels * pixel_size  # RMS = sigma for Gaussian

    print("\nTest parameters:")
    print("  Image size: {0}x{1} pixels".format(size, size))
    print("  Gaussian sigma: {0} pixels = {1:.4f} mm".format(sigma_pixels, sigma_pixels * pixel_size))
    print("  Pixel size: {0} mm".format(pixel_size))

    print("\nCalculation results:")
    print("  Centroid X: {0:.17g} mm (expected: {1:.17g} mm)".format(centroid_x, expected_centroid))
    print("  Centroid Y: {0:.17g} mm (expected: {1:.17g} mm)".format(centroid_y, expected_centroid))
    print("  RMS X: {0:.17g} mm (expected: {1:.17g} mm)".format(rms_x, expected_rms))
    print("  RMS Y: {0:.17g} mm (expected: {1:.17g} mm)".format(rms_y, expected_rms))

    print("\nDeviations from theoretical values:")
    print("  Centroid X: {0:.17g} mm".format(abs(centroid_x - expected_centroid)))
    print("  Centroid Y: {0:.17g} mm".format(abs(centroid_y - expected_centroid)))
    print("  RMS X: {0:.17g} mm ({1:.17g}%)".format(abs(rms_x - expected_rms), abs(rms_x - expected_rms)/expected_rms*100))
    print("  RMS Y: {0:.17g} mm ({1:.17g}%)".format(abs(rms_y - expected_rms), abs(rms_y - expected_rms)/expected_rms*100))

    # Check accuracy (allowable error 0.1%)
    assert abs(centroid_x) < 0.0001, "Centroid X should be ~0, got {0}".format(centroid_x)
    assert abs(centroid_y) < 0.0001, "Centroid Y should be ~0, got {0}".format(centroid_y)
    assert abs(rms_x - expected_rms) / expected_rms < 0.001, \
        "RMS X should be ~{0}, got {1}".format(expected_rms, rms_x)
    assert abs(rms_y - expected_rms) / expected_rms < 0.001, \
        "RMS Y should be ~{0}, got {1}".format(expected_rms, rms_y)

    print("\n* Test passed! Centroid and RMS match theoretical values.")

    return image, centroid_x, centroid_y, rms_x, rms_y, sigma_pixels, pixel_size


def test_gaussian_offset():
    """
    Test 2: Gaussian distribution with offset.
    Centroid should coincide with the center of distribution.
    RMS is independent of offset and should be equal to sigma.
    """
    print("\n" + "="*70)
    print("Test 2: Offset Gaussian distribution")
    print("="*70)

    analyzer = ImageAnalyzer()

    size = 101
    sigma_pixels = 8.0
    pixel_size = 0.0284
    offset_x = 5.0  # Offset in pixels
    offset_y = -3.0

    # Create Gaussian with offset
    image = create_2d_gaussian(size, sigma_pixels, center=(offset_x, offset_y))

    # Calculate centroid and RMS
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    # Expected values
    expected_centroid_x = offset_x * pixel_size
    expected_centroid_y = offset_y * pixel_size
    expected_rms = sigma_pixels * pixel_size

    print("\nTest parameters:")
    print("  Image size: {0}x{1} pixels".format(size, size))
    print("  Gaussian sigma: {0} pixels = {1:.4f} mm".format(sigma_pixels, sigma_pixels * pixel_size))
    print("  Offset: ({0}, {1}) pixels = ({2:.4f}, {3:.4f}) mm".format(offset_x, offset_y, expected_centroid_x, expected_centroid_y))

    print("\nCalculation results:")
    print("  Centroid X: {0:.17g} mm (expected: {1:.17g} mm)".format(centroid_x, expected_centroid_x))
    print("  Centroid Y: {0:.17g} mm (expected: {1:.17g} mm)".format(centroid_y, expected_centroid_y))
    print("  RMS X: {0:.17g} mm (expected: {1:.17g} mm)".format(rms_x, expected_rms))
    print("  RMS Y: {0:.17g} mm (expected: {1:.17g} mm)".format(rms_y, expected_rms))

    print("\nDeviations from theoretical values:")
    print("  Centroid X: {0:.17g} mm".format(abs(centroid_x - expected_centroid_x)))
    print("  Centroid Y: {0:.17g} mm".format(abs(centroid_y - expected_centroid_y)))
    print("  RMS X: {0:.17g} mm ({1:.17g}%)".format(abs(rms_x - expected_rms), abs(rms_x - expected_rms)/expected_rms*100))
    print("  RMS Y: {0:.17g} mm ({1:.17g}%)".format(abs(rms_y - expected_rms), abs(rms_y - expected_rms)/expected_rms*100))

    # Check accuracy
    assert abs(centroid_x - expected_centroid_x) < 0.001, \
        "Centroid X should be ~{0}, got {1}".format(expected_centroid_x, centroid_x)
    assert abs(centroid_y - expected_centroid_y) < 0.001, \
        "Centroid Y should be ~{0}, got {1}".format(expected_centroid_y, centroid_y)
    assert abs(rms_x - expected_rms) / expected_rms < 0.001, \
        "RMS X should be ~{0}, got {1}".format(expected_rms, rms_x)
    assert abs(rms_y - expected_rms) / expected_rms < 0.001, \
        "RMS Y should be ~{0}, got {1}".format(expected_rms, rms_y)

    print("\n* Test passed! Centroid matches offset, RMS is independent of offset.")

    return image, centroid_x, centroid_y, rms_x, rms_y, sigma_pixels, pixel_size, offset_x, offset_y


def test_gaussian_elliptical():
    """
    Test 3: Elliptical Gaussian distribution.
    Different sigma for X and Y axes.
    """
    print("\n" + "="*70)
    print("Test 3: Elliptical Gaussian distribution")
    print("="*70)

    analyzer = ImageAnalyzer()

    size = 101
    sigma_x = 12.0  # Different sigma for axes
    sigma_y = 6.0
    pixel_size = 0.0284

    # Create elliptical Gaussian manually
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)

    image = np.exp(-(xx**2 / (2*sigma_x**2) + yy**2 / (2*sigma_y**2)))

    # Calculate centroid and RMS
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    # Expected values
    expected_centroid = 0.0
    expected_rms_x = sigma_x * pixel_size
    expected_rms_y = sigma_y * pixel_size

    print("\nTest parameters:")
    print("  Image size: {0}x{1} pixels".format(size, size))
    print("  Sigma X: {0} pixels = {1:.4f} mm".format(sigma_x, sigma_x * pixel_size))
    print("  Sigma Y: {0} pixels = {1:.4f} mm".format(sigma_y, sigma_y * pixel_size))
    print("  Axis ratio: {0:.2f}:1".format(sigma_x/sigma_y))

    print("\nCalculation results:")
    print("  Centroid X: {0:.17g} mm (expected: {1:.17g} mm)".format(centroid_x, expected_centroid))
    print("  Centroid Y: {0:.17g} mm (expected: {1:.17g} mm)".format(centroid_y, expected_centroid))
    print("  RMS X: {0:.17g} mm (expected: {1:.17g} mm)".format(rms_x, expected_rms_x))
    print("  RMS Y: {0:.17g} mm (expected: {1:.17g} mm)".format(rms_y, expected_rms_y))

    print("\nDeviations from theoretical values:")
    print("  Centroid X: {0:.17g} mm".format(abs(centroid_x - expected_centroid)))
    print("  Centroid Y: {0:.17g} mm".format(abs(centroid_y - expected_centroid)))
    print("  RMS X: {0:.17g} mm ({1:.17g}%)".format(abs(rms_x - expected_rms_x), abs(rms_x - expected_rms_x)/expected_rms_x*100))
    print("  RMS Y: {0:.17g} mm ({1:.17g}%)".format(abs(rms_y - expected_rms_y), abs(rms_y - expected_rms_y)/expected_rms_y*100))

    # Check accuracy
    assert abs(centroid_x) < 0.0001, "Centroid X should be ~0, got {0}".format(centroid_x)
    assert abs(centroid_y) < 0.0001, "Centroid Y should be ~0, got {0}".format(centroid_y)
    assert abs(rms_x - expected_rms_x) / expected_rms_x < 0.001, \
        "RMS X should be ~{0}, got {1}".format(expected_rms_x, rms_x)
    assert abs(rms_y - expected_rms_y) / expected_rms_y < 0.001, \
        "RMS Y should be ~{0}, got {1}".format(expected_rms_y, rms_y)

    print("\n* Test passed! Elliptical distribution processed correctly.")

    return image, centroid_x, centroid_y, rms_x, rms_y, sigma_x, sigma_y, pixel_size


def visualize_test(image, centroid_x, centroid_y, rms_x, rms_y,
                   expected_centroid_x, expected_centroid_y, expected_rms_x, expected_rms_y,
                   pixel_size, title, filename):
    """
    Visualization of test results for visual validation.
    """
    from matplotlib.patches import Ellipse

    analyzer = ImageAnalyzer()

    # Get projections
    x_coords, x_proj, y_coords, y_proj = analyzer.calculate_projections(
        image, pixel_size, pixel_size
    )

    size = image.shape[0]

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 2D image
    ax = axes[0, 0]
    extent = [
        -(size-1)/2 * pixel_size,
        (size-1)/2 * pixel_size,
        -(size-1)/2 * pixel_size,
        (size-1)/2 * pixel_size
    ]
    im = ax.imshow(image, extent=extent, origin='lower', cmap='jet')
    ax.axhline(y=centroid_y, color='cyan', linestyle='--',
               label='Centroid Y={0:.4f}'.format(centroid_y))
    ax.axvline(x=centroid_x, color='cyan', linestyle='--',
               label='Centroid X={0:.4f}'.format(centroid_x))

    # RMS boundary (1 sigma)
    ellipse = Ellipse((centroid_x, centroid_y), 2*rms_x, 2*rms_y,
                     fill=False, edgecolor='lime', linestyle='--', linewidth=2,
                     label='1sigma: ({0:.4f}, {1:.4f})'.format(rms_x, rms_y))
    ax.add_patch(ellipse)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('2D Gaussian Distribution')
    ax.legend()
    plt.colorbar(im, ax=ax)

    # X projection
    ax = axes[0, 1]
    ax.plot(x_coords, x_proj, 'b-', linewidth=2, label='X projection')
    ax.axvline(x=centroid_x, color='red', linestyle='--',
               label='Centroid={0:.4f}'.format(centroid_x))
    ax.axvline(x=centroid_x - rms_x, color='orange', linestyle=':',
               label='RMS={0:.4f}'.format(rms_x))
    ax.axvline(x=centroid_x + rms_x, color='orange', linestyle=':')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Projection on X axis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y projection
    ax = axes[1, 0]
    ax.plot(y_coords, y_proj, 'g-', linewidth=2, label='Y projection')
    ax.axvline(x=centroid_y, color='red', linestyle='--',
               label='Centroid={0:.4f}'.format(centroid_y))
    ax.axvline(x=centroid_y - rms_y, color='orange', linestyle=':',
               label='RMS={0:.4f}'.format(rms_y))
    ax.axvline(x=centroid_y + rms_y, color='orange', linestyle=':')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Projection on Y axis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis('off')

    info_text = """
    ANALYSIS RESULTS
    {0}

    Image parameters:
    * Size: {1}x{2} pixels
    * Pixel size: {3} mm
    * Expected centroid: ({4:.4f}, {5:.4f}) mm
    * Expected RMS: ({6:.4f}, {7:.4f}) mm

    Centroid (center of mass):
    * X: {8:.17g} mm
    * Y: {9:.17g} mm

    RMS sizes (beam width):
    * X: {10:.17g} mm
    * Y: {11:.17g} mm

    Accuracy (deviation from theory):
    * Centroid X: {12:.17g} mm
    * Centroid Y: {13:.17g} mm
    * RMS X: {14:.17g}%
    * RMS Y: {15:.17g}%

    * Calculations are correct!
    """.format(
        '='*40,
        size, size,
        pixel_size,
        expected_centroid_x, expected_centroid_y,
        expected_rms_x, expected_rms_y,
        centroid_x, centroid_y,
        rms_x, rms_y,
        abs(centroid_x - expected_centroid_x),
        abs(centroid_y - expected_centroid_y),
        abs(rms_x - expected_rms_x) / expected_rms_x * 100 if expected_rms_x != 0 else 0,
        abs(rms_y - expected_rms_y) / expected_rms_y * 100 if expected_rms_y != 0 else 0
    )

    ax.text(0.1, 0.5, info_text, fontsize=9, family='monospace',
           verticalalignment='center')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("\n* Plot saved to {0}".format(filename))

    return fig


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  EXTENDED VALIDATION OF CENTROID AND RMS CALCULATIONS  ".center(70))
    print("="*70)

    try:
        # Run all tests
        print("\n>>> Running Test 1: Centered Gaussian")
        image1, cx1, cy1, rx1, ry1, sigma1, psize1 = test_gaussian_centered()

        print("\n>>> Running Test 2: Offset Gaussian")
        image2, cx2, cy2, rx2, ry2, sigma2, psize2, ox2, oy2 = test_gaussian_offset()

        print("\n>>> Running Test 3: Elliptical Gaussian")
        image3, cx3, cy3, rx3, ry3, sx3, sy3, psize3 = test_gaussian_elliptical()

        # Visualization
        print("\n" + "="*70)
        print("Creating visualizations...")
        print("="*70)

        try:
            # Visualize Test 1
            visualize_test(
                image1, cx1, cy1, rx1, ry1,
                0.0, 0.0, sigma1 * psize1, sigma1 * psize1,
                psize1,
                "Test 1: Centered Gaussian Distribution",
                'd:/DEV/CANDLE/Beam-Profile-Analyzer/gaussian_test1_centered.png'
            )

            # Visualize Test 2
            visualize_test(
                image2, cx2, cy2, rx2, ry2,
                ox2 * psize2, oy2 * psize2, sigma2 * psize2, sigma2 * psize2,
                psize2,
                "Test 2: Offset Gaussian Distribution",
                'd:/DEV/CANDLE/Beam-Profile-Analyzer/gaussian_test2_offset.png'
            )

            # Visualize Test 3
            visualize_test(
                image3, cx3, cy3, rx3, ry3,
                0.0, 0.0, sx3 * psize3, sy3 * psize3,
                psize3,
                "Test 3: Elliptical Gaussian Distribution",
                'd:/DEV/CANDLE/Beam-Profile-Analyzer/gaussian_test3_elliptical.png'
            )

            print("\nVisual verification:")
            print("  - Centroid should be at the distribution center (cyan lines)")
            print("  - RMS contour (green) should encompass ~68% of beam energy")

        except ImportError:
            print("\n! matplotlib is not installed - visualization skipped")
            print("  Install with: pip install matplotlib")

        print("\n" + "="*70)
        print("  *** ALL TESTS PASSED SUCCESSFULLY! ***  ".center(70))
        print("  Centroid and RMS calculations are fully correct  ".center(70))
        print("  and match theoretical values!  ".center(70))
        print("="*70)

    except AssertionError as e:
        print("\n* ERROR: {0}".format(e))
        print("\nCheck the implementation of calculations!")
        raise
