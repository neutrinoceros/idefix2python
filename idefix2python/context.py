from pathlib import Path
import os
from .tools import LOG
import argparse
from . import tools
from .vtk_io import readVTK
import numpy as np

CARTESIAN_DIMENSION_NAMES = {
    "cartesian": [r"$x$", r"$y$", r"$z$"],
    "polar": [r"$x$", r"$y$", r"$z$"],
    "cylindrical": [r"$x$", r"$z$", None],
    "spherical": [r"$x$", r"$z$", r"$y$"],
}

DIMENSION_NAMES = {
    "cartesian": [r"$x$", r"$y$", r"$z$"],
    "polar": [r"$r$", r"$\phi$", r"$z$"],
    "cylindrical": [r"$r$", r"$z$", None],
    "spherical": [r"$r$", r"$\theta$", r"$\phi$"],
}


def _get_args():
    """Builds the default command-line argument parser for Idefix2Python."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--frame",
        nargs="*",
        default=None,
        help="integer: will only render this frame",
        type=int,
        dest="doOnlyFrames",
    )

    parser.add_argument(
        "--no-bounds",
        action="store_true",
        dest="noBounds",
        help="will ignore the config file and let free bounds on colorbars",
    )

    parser.add_argument(
        "-om", action="store_true", help="only movie?", dest="onlyMovie"
    )

    parser.add_argument(
        "-oa", action="store_true", help="only analysis?", dest="onlyAnalysis"
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of CPU cores to use",
    )

    parser.add_argument(
        "-u",
        "--until",
        type=lambda s: int(s) if s.isdigit() else float(s),
        default=1,
        help="To read only a part of the data. float between 0 and 1 is interpreted as a fraction, int as an output number, and a float > 1 as a time.",
        dest="until",
    )

    parser.add_argument(
        "-e",
        "--every",
        type=int,
        default=1,
        help="Read every Nth output file (N>=1). For example, -e 2 reads every second file.",
    )

    args = parser.parse_args()
    if args.doOnlyFrames is None:
        args.doOnlyFrames = False
    elif len(args.doOnlyFrames) == 0:
        args.doOnlyFrames = [0]

    return args


class OutputTypeInfo:
    """
    Different types of output: global (vtk), slice (vtk), timevol (dat), particles (vtk)
    """

    def __init__(self, name, files):
        self.name = name
        self.files = files
        self.geometry = None
        self.dimensions = None

        self.dataHas = {
            "Pressure": False,
            "B": False,
            "Dust": False,
            "Particles": False,
        }

        if len(self.files) > 0:
            self.status = True
            self.test_file = self.files[0]
            self.ext = self.test_file.suffixes[-1]

            self._set_testData()
            self.get_availableKeys()

        else:
            self.status = False

    def _set_testData(self):
        if "vtk" in self.ext:
            vtk = readVTK(self.test_file)
            self.testData = vtk.data
            self.geometry = vtk.geometry
            self.vtk = vtk
            if "part" not in str(self.test_file):
                self.dimensions = vtk.dimensions

        elif "dat" in self.ext:
            self.testData = tools.dat_to_dict(self.test_file)

    def get_availableKeys(self):
        if not self.status:
            return f"No {self.name} present"
        LOG(f"------ Available fields in {self.test_file} ------")

        for qt in self.testData.keys():
            LOG(f"{qt:>10} {np.shape(self.testData[qt].data)}")
            if qt == "PRS":
                self.dataHas["Pressure"] = True
            elif qt.startswith("BX"):
                self.dataHas["B"] = True
            elif qt.startswith("Dust"):
                self.dataHas["Dust"] = True
            elif qt.startswith("PART"):
                self.dataHas["Particles"] = True
        if "vtk" in self.ext:
            dataStart = self.files[0]
            dataEnd = self.files[-1]  # TODO end option
            self.tStart = readVTK(dataStart).t[0]
            self.tEnd = readVTK(dataEnd).t[0]

        elif "dat" in self.ext:
            raise NotImplementedError()


class RunContext:
    """
    The first thing to initiate.

    Handles data location and directory creation. Detects simulation geometry,
    dimensions, and available fields.

    Args:
        runName (str): The unique name of the run.
        projectPath (str | Path, optional): The root directory of the project.
            Defaults to the current directory (".").
        **kwargs: Additional optional parameters:

            * configPath (str | Path): Path to a specific configuration file.
            * partFolder (str): Folder path containing the particles data.
            * frameFolder (str): Folder name where the rendered frames will be stored.
            * active_directions (list): List of active coordinate directions.
            * debug (bool): debug mode will show the .ini file.
                Defaults to False.
            * iniPath (Path): Custom path to the .ini input file. Defaults to
              `projectPath/inputs/{runName}.ini`.

    Note:
        The expected location for the .vtk files is `projectPath/outputs/runName/vtks`.
        By default, the rendered frame will be located in `projectPath/frames/runName`.
    """

    def __init__(self, runName, projectPath=".", **kwargs):
        self.runName = runName
        self.projectPath = Path(projectPath)
        self.projectPath.resolve(strict=True)

        self.debug = kwargs.get("debug", False)

        self.userArgs = kwargs.get("args", _get_args())

        self.config = {}
        configPath = kwargs.get("configPath", None)
        self.configPath = configPath
        if configPath is not None:
            LOG(f"config.json file requested: {configPath}")
            self.config = tools.process_configs(configPath)

        self.dataPath = self.projectPath / "outputs" / runName
        self.iniPath = Path(
            kwargs.get("iniPath", self.projectPath / "inputs" / f"{runName}.ini")
        )
        self.format_inputs_text = ""
        if self.debug:
            if self.iniPath.is_file():
                self.format_inputs_text = tools.formatInputs(self.iniPath)
            else:
                raise FileNotFoundError(
                    f"debug requested but {self.iniPath} doesn't exist"
                )

        self.partFolder = kwargs.get("partFolder", None)

        self.frameFolderName = kwargs.get("frameFolder", runName)
        self.active_directions = kwargs.get("active_directions", [])
        # for part*.vtk, the readVTK routine can't deduce the number of dimensions.
        # The context will try to find the dimensions in data*.vtk
        # If there is no data*.vtk, the user has to pass the dimensions.

        self._setup_directories()
        self._check_data()

        self.gridInfo = GridInfo(self)

    def _setup_directories(self):
        self.frameRootFolder = self.projectPath / "frames" / self.frameFolderName
        self.globalFolder = self.frameRootFolder / "global"
        self.slice1Folder = self.frameRootFolder / "slice1"
        self.videosFolder = self.projectPath / "videos"

        for path in [
            self.globalFolder,
            self.slice1Folder,
            self.videosFolder,
        ]:
            os.makedirs(path, exist_ok=True)
            # except OSError as _:
            # pass
            # subfolder = os.path.basename(path)
            # content = glob.glob(f"{path}/*")
            # user_agree = input(
            #     f"Will overwrite the {subfolder} folder ({len(content)} files) [o/r/n] (overwrite, remove, no)"
            # )
            # if user_agree == "r":
            #     for f in content:
            #         os.remove(f)
            # elif user_agree == "n":
            #     exit()

    def _check_data(self):
        "Show fields in every kind of data and detect is there are Pressure, B, Dust or Particles fields. Also detects the geometry. Also detect t_start and t_end"
        self.outputTypes_info = {}
        # self.outputTypes = ["analysis", "slice1", "vtk", "particles"]
        self.outputTypes = ["slice1", "vtk", "particles"]
        # self.outputTypes_info["analysis"] = OutputTypeInfo(self.analysis_path, "analysis")
        self.outputTypes_info["vtk"] = OutputTypeInfo("vtk", self.get_global_vtkFiles())
        self.outputTypes_info["slice1"] = OutputTypeInfo(
            "slice1", self.get_slice1_vtkFiles()
        )
        self.outputTypes_info["particles"] = OutputTypeInfo(
            "particles", self.get_particles_vtkFiles()
        )
        self.outputTypes_info["particles"].dimensions = self.outputTypes_info[
            "vtk"
        ].dimensions
        # There's no way to deduce the number of dimensions from the part*.vtk files but it has to be the same as in the global vtk

        if (
            self.partFolder is not None
            and not self.outputTypes_info["particles"].status
        ):
            raise FileNotFoundError(
                f"the folder {self.partFolder} doesn't seem to contain any part*vtk"
            )

        ## Everything is deduced from the global vtk
        vtkInfo = self.outputTypes_info["vtk"]
        partInfo = self.outputTypes_info["particles"]
        if vtkInfo.status:
            geometry = vtkInfo.geometry
        elif partInfo.status:
            geometry = partInfo.geometry
        else:
            raise Exception("No vtk files were found?")

        if len(self.active_directions) == 0:
            if not vtkInfo.status:
                raise Exception(
                    "No data*.vtk detected. Please provide active_directions."
                )
            vtk = vtkInfo.vtk
            for direction, ncell in enumerate([vtk.nx, vtk.ny, vtk.nz]):
                if ncell > 1:
                    self.active_directions.append(direction)

        dimensions = len(self.active_directions)
        self.geometry = geometry
        self.dimensions = dimensions

        self.active_directions_labels = [
            DIMENSION_NAMES[self.geometry][dir] for dir in self.active_directions
        ]
        LOG("Dimensions detected: ", self.dimensions)
        LOG("Active axes", self.active_directions_labels)

        if self.outputTypes_info["particles"].status:
            self.all_particles_uids = self.outputTypes_info["particles"].testData["uid"]
            self.particles_nb = len(self.all_particles_uids)
            LOG(f"Particles detected: {self.particles_nb}")
        else:
            self.particles_nb = 0
            self.all_particles_uids = []

    def _get_lastfile_to_read(self, filelist):
        """
        expects a sorted list
        """
        until = self.userArgs.until
        if len(filelist) == 0:
            lastframe = -1
        elif 0 <= until <= 1:
            lastframe = int(len(filelist) * until)
        elif isinstance(until, int):
            lastframe = until
        elif isinstance(until, float):
            if str(filelist[-1]).endswith(".vtk"):
                tend = readVTK(filelist[-1]).t[0]
                tstart = readVTK(filelist[0]).t[0]
                if until < tstart:
                    raise Exception(
                        f"Value of until ({until}) is inferior than the first file time"
                    )
                lastframe = int((until - tstart) / (tend - tstart) * len(filelist))
                if lastframe + 1 < len(filelist):
                    lastframe += 1
        return lastframe

    def get_global_vtkFiles(self):
        pattern = "vtks/data*.vtk"
        filelist = sorted(self.dataPath.glob(pattern))
        lastfile = self._get_lastfile_to_read(filelist)
        filelist = filelist[:lastfile]
        return filelist[:: self.userArgs.every]

    def get_slice1_vtkFiles(self):
        pattern = "vtks/slice1*.vtk"
        filelist = sorted(self.dataPath.glob(pattern))
        lastfile = self._get_lastfile_to_read(filelist)
        filelist = filelist[:lastfile]
        return filelist[:: self.userArgs.every]

    def get_particles_vtkFiles(self):
        if self.partFolder is not None:
            filelist = sorted(Path(self.partFolder).glob("part*.vtk"))
        else:
            pattern = "vtks/part*.vtk"
            filelist = sorted(self.dataPath.glob(pattern))

        lastfile = self._get_lastfile_to_read(filelist)
        filelist = filelist[:lastfile]
        return filelist[:: self.userArgs.every]


class GridInfo:
    def __init__(self, context):
        self.context = context
        self.geometry = context.geometry
        self.dimensions = context.dimensions
        self.grid_name_1, self.grid_name_2 = self.get_cartesian_grid_labels()
        self.axis_name_1, self.axis_name_2 = self.get_native_grid_labels()
        self.shape = None
        if self.context.outputTypes_info["vtk"].status:
            self.X1Line, self.X2Line = self.get_grid_line_points()
            if self.context.dimensions == 1:
                self.xmin = np.min(self.X1Line)
                self.xmax = np.max(self.X1Line)
                self.shape = np.shape(self.X1Line)
            elif self.context.dimensions == 2:
                # Regardless of the geometry, we need the cartesian grid (X,Y,Z) for pcolormesh
                self.X1, self.X2 = np.meshgrid(self.X1Line, self.X2Line)
                self.grid1, self.grid2 = tools.convertGrid_toXZ(
                    self.X1, self.X2, self.context.geometry
                )

                self.xmin = 0  # works good atm
                self.xmax = np.max(self.grid1)
                self.ymax = np.max(self.grid2)
                self.ymin = np.min(self.grid2)

                self.shape = np.shape(self.X1)

    def get_cartesian_grid_labels(self):
        # 2D fields are always showed in cartesian. Thus, the labels should be cartesian.
        names = [None, None]
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                names[i] = CARTESIAN_DIMENSION_NAMES[self.context.geometry][dir]

        return names

    def get_native_grid_labels(self):
        names = [None, None]
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                names[i] = DIMENSION_NAMES[self.context.geometry][dir]
        return names

    def get_grid_line_points(self):
        Lines = [None, None]

        vtk = self.context.outputTypes_info["vtk"].vtk
        for i, dir in enumerate(self.context.active_directions):
            if i < 2:
                # max 2 dimensions is supported
                Lines[i] = tools.get_Position(vtk, self.context.geometry, dir)

        return Lines

    def apply_zoom(self, zoom):
        if zoom is None:
            self.grid1_toshow, self.grid2_toshow = self.grid1, self.grid2
            self.X1Line_toshow, self.X2Line_toshow = self.X1Line, self.X2Line
            self.mask1 = np.full(self.X1Line.shape, True, dtype=bool)
            self.mask2 = np.full(self.X2Line.shape, True, dtype=bool)
        else:
            print(np.shape(self.X1Line), np.shape(self.X2Line))
            self.mask1, self.mask2 = zoom(self.X1Line, self.X2Line)
            self.X1Line_toshow = self.X1Line[self.mask1]
            self.X2Line_toshow = self.X2Line[self.mask2]
            self.grid1_toshow, self.grid2_toshow = np.meshgrid(
                self.X1Line_toshow, self.X2Line_toshow
            )
        # self.mask, _ = np.meshgrid(self.mask1, self.mask2)
        self.mask = np.logical_and.outer(self.mask2, self.mask1)
        print("hey", self.mask1)
        print("hey", self.mask2)
        print("hey", self.mask)
