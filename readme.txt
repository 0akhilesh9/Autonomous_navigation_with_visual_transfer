Visual Transfer Learning for Autonomous Navigation Task:

A pre-trained model (Alexnet) is fine tuned on the images sampled from the scenes and trained for basic vision tasks like edge detection, contour detection. The performance of this fine-tuned model is compared against the basic model (not fine-tuned).


Models tried:

DQN and PPO


Prerequisites:

habitat-sim and habitat-lab



To maually navigate through a scene:

cd into 'habitat-sim' folder and run:
		build/viewer <path_to_scene.glb>
		ex: build/viewer /home/userone/workspace/bed1/project/data/gibson/Albertville.glb


Data setup:

Download gibson_habitat_trainval.zip from Gibson dataset page (https://github.com/StanfordVL/GibsonEnv/tree/master/gibson/data)
-Download 3DSceneGraph_medium.zip from 3DSceneGraph page (https://3dscenegraph.stanford.edu/database.html)
-Download gibson_medium.tar.gz from Gibson dataset page for mesh.obj files
			(Under Gibson Env V1 data:
					"Tiny" Partition (8.02 GB): gibson_tiny.tar.gz
					"Medium" Partition (20.8 GB): gibson_medium.tar.gz)

-Run the 'gen_gibson_semantics.sh' script. For this script to run properly every scene needs to have .glb, .navmesh, .npz, mesh.obj files
-Also, while installing habita_sim (i.e., while running python setup.py install) you need to specify "--build-datatool" flag to install data tool as this will be needed for generating semantic information from Gibson dataset
ex:
	python setup.py --build-datatool install


How to run:
cd into 'habita_sim' directory and run:
	tools/gen_gibson_semantics.sh <path_to/3DSceneGraph_medium/automated_graph>  <path_to/GibsonDataset>  <path_to/output>
ex:
	tools/gen_gibson_semantics.sh     /home/userone/workspace/bed1/project/data/3DSceneGraph_medium/automated_graph     /home/userone/workspace/bed1/project/data/gibson      /home/userone/workspace/bed1/project/data/gibson_sem/

Check for any errors during script execution.

Sample output:

	Albertville
	wrote 3297222 bytes
	wrote 64751 bytes
	WARNING: Logging before InitGoogleLogging() is written to STDERR
	I1020 06:26:48.036468  8682 datatool.cpp:44] createGibsonSemanticMesh
	I1020 06:26:49.362576  8682 datatool.cpp:182] task: "create_gibson_semantic_mesh" done


Few scenes that have all the files:  (gibson_habitat.zip from gibson database, gibson_medium.tar.gz, 3DSceneGraph_medium.zip)
Albertville.glb
Goffs.glb
Hainesburg.glb
Micanopy.glb
Nuevo.glb
Oyens.glb
Pablo.glb
Rosser.glb
Sands.glb



https://github.com/facebookresearch/habitat-sim