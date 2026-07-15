"""Language-conditioned Vision-Language-Action manipulation stack for the SO-101.

Subpackages:
    data       - dataset loading, normalization stats, layout randomization, paraphrases
    models     - the behavior-cloning baseline (CLIP text + ResNet image + MLP head)
    training   - training loops and config plumbing
    inference  - policy server, robot client, action chunking, safety filter
    eval       - trial-spec evaluation harness and statistics
    robot      - ROS2 controller and SO-101 forward kinematics
    utils      - seeding and logging helpers
"""

__version__ = "0.1.0"
