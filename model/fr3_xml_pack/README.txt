fr3_xml_pack
===========

Files
-----
- fr3_push_modular_compsim.xml : main wrapper that includes everything
- fr3.xml                      : Franka FR3 model (expects an 'assets/' folder next to it)
- t_block_optimized.xml         : T-block + floor plane (visual)
- table_worldbody.xml           : simple visible table (visual)
- obstacles_worldbody_fixed.xml : obstacles + mocap markers

Important note about physics
----------------------------
Your compsim scripts usually disable MuJoCo contacts and solve contact in Python (ground MCP + tool contact).
So the table here is VISUAL ONLY unless you also change the ground model in Python.

How to use
----------
1) Put these XML files in the SAME directory.
2) Make sure you also have the robot meshes/textures under: ./assets/
   (fr3.xml uses <compiler meshdir="assets"/>).
3) Point your script to the wrapper XML, e.g.:

   python3 -m scripts.push_fr3_waypoints_compsim_live \
       --xml ./fr3_push_modular_compsim.xml --live_view
