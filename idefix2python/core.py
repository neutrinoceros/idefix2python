from multiprocessing import Pool
from itertools import repeat

from .tools import LOG
from .renderer import SliceRenderer
from .processor import PhysicsProcessor, PartsInfo
from .quantities import (
    PartQuantity,
    SpaceTimeHeatmap,
    MapMovie2D,
    LineMovie1D,
    OneComponentOneVariable,
)
from .tools import convertGrid_toXZ


class Pipeline:
    def __init__(
        self,
        Context,
        figs,
        streamLines=None,
        **options,
    ):
        """
        Coordinates the detection, processing, and rendering of the simulation data.
        :param Context: The RunContext object containing simulation metadata.
        :type Context: RunContext
        :param figs: `Figure` instances.
        :type figs: list[Figure]
        :param streamLines: Configuration for streamlines overlays.
        :type streamLines: StreamlineConfig, optional
        **options: Additional optional parameters:
            * scatter_particles (bool): only scatter particles positions instead of the whole trajectory on MapMovie2D
            * zoom (callable): Zoom level for the rendering view (for 2D only currently). For example, `zoom(x1,x2): return x1, x2<np.pi/2`.
        """
        self.context = Context
        self.userArgs = self.context.userArgs

        self.streamLines = streamLines

        self.processor = PhysicsProcessor(self.context, self.userArgs, self.streamLines)

        self.figs = figs
        self.mapmovies2D = []
        self.linemovies1D = []
        self.spaceTimeHeatmaps = []
        self.oneC_oneVs = []
        self.partQuantities = []
        self.quantities = []

        qty_tocompute = []
        self.particles_requested = False

        self.options = options
        for fig in figs:
            for qtyInfo in fig.quantities:
                if isinstance(qtyInfo, PartQuantity):
                    self.partQuantities.append(qtyInfo)
                elif isinstance(qtyInfo, SpaceTimeHeatmap):
                    self.spaceTimeHeatmaps.append(qtyInfo)
                elif isinstance(qtyInfo, LineMovie1D):
                    self.linemovies1D.append(qtyInfo)
                elif isinstance(qtyInfo, MapMovie2D):
                    self.mapmovies2D.append(qtyInfo)
                elif isinstance(qtyInfo, OneComponentOneVariable):
                    self.oneC_oneVs.append(qtyInfo)
                self.quantities.append(qtyInfo)

                if qtyInfo.compute is not None:
                    # TODO add safeguard in case two identical keys with differents compute.
                    qty_tocompute.append(qtyInfo)

                if qtyInfo.uids is not None:
                    self.particles_requested = True
                    if qtyInfo.uids == "all":
                        qtyInfo.uids = self.context.all_particles_uids

        self.processor.partsInfo = PartsInfo(
            self.context.active_directions
        )  # pipeline bro helping clueless processor

        if self.particles_requested:
            self.partQuantities += self.processor.partsInfo.global_partsqty_togather

        self.processor.set_qty_tocompute(qty_tocompute)
        self.processor.set_partQuantities(self.partQuantities)

        self._name_frames()

    def _check_everything_alright(self):
        """
        Simply check if necessary files exist.
        """
        # Check whether the particles requested exist
        available_uids = set(self.context.all_particles_uids)
        for qty in [
            *self.partQuantities,
            *self.spaceTimeHeatmaps,
            *self.linemovies1D,
            *self.mapmovies2D,
        ]:
            if isinstance(qty.uids, list) and len(qty.uids) > 0:
                missing_uids = set(qty.uids) - available_uids
                if len(missing_uids) > 0:
                    raise Exception(
                        f"One or more requested particle uids do not exist: {sorted(missing_uids)}"
                    )

        partInfo = self.context.outputTypes_info["particles"]
        globalInfo = self.context.outputTypes_info["vtk"]
        if len(self.partQuantities) > 0 or self.particles_requested:
            if not partInfo.status:
                raise Exception(
                    f"Particle quantities were requested, but no part*.vtk files were found at {partInfo.files}"
                )

        if (
            len(self.spaceTimeHeatmaps) > 0
            or len(self.linemovies1D)
            or len(self.mapmovies2D) > 0
        ):
            if not globalInfo.status:
                raise Exception(
                    f"Global quantities were requested, but no data*.vtk files were found at {globalInfo.files}"
                )

        LOG("Quantities to compute:")
        LOG(f"{'LineMovie1D':>20}: {len(self.linemovies1D)}")
        LOG(f"{'MapMovie2D':>20}: {len(self.mapmovies2D)}")
        LOG(f"{'SpaceTimeHeatmap':>20}: {len(self.spaceTimeHeatmaps)}")
        LOG(f"{'PartQuantity':>20}: {len(self.partQuantities)}")

    def run(self):
        """
        Pray.
        """

        self.renderer = SliceRenderer(
            self.context,
            self.processor,
            self.figs,
            self.userArgs,
            self.options,
        )

        # -om -> Only renders Movie
        if self.userArgs.onlyMovie:
            for fig in self.figs:
                if fig.movie:
                    self.renderer.render_movie(fig)
            LOG("Only movie requested. Godspeed.")
            return

        self._check_everything_alright()
        remaining_fields_tobound = self._apply_config()

        # Data to gather : 1C1V (including parts) and spheatmaps.
        # Gather data will be stored in a dict. Each key correspond to
        # one given quantity.
        vtktimes = None
        keys_togather = set()
        keys_tobound = remaining_fields_tobound
        for qty in self.oneC_oneVs:
            if qty.xqty is not None:
                keys_togather.add(qty.xqty)

        for qty in self.partQuantities + self.spaceTimeHeatmaps + self.oneC_oneVs:
            if qty.key not in keys_togather:
                keys_togather.add(qty)

        if len(keys_togather) > 0 or len(keys_tobound) > 0:
            LOG("Gathering data and/or bounds, please wait...")
            files_diff = len(self.vtkList) - len(self.partList)
            if files_diff > 0:
                vtkList_extended = self.vtkList
                partList_extended = self.partList + [None] * files_diff
            elif files_diff < 0:
                vtkList_extended = self.vtkList + [None] * (-files_diff)
                partList_extended = self.partList
            else:
                vtkList_extended = self.vtkList
                partList_extended = self.partList

            with Pool(self.userArgs.jobs) as pool:
                gathered_data_and_bounds = pool.starmap(
                    self.processor.gather_1Cquantities,
                    zip(
                        vtkList_extended,
                        partList_extended,
                        repeat(keys_togather),
                        repeat(keys_tobound),
                    ),
                )

            nb_vtks = len(gathered_data_and_bounds)

            # Let's browser that huge result
            # Redistribute all the gathered data to all quantities.
            vtktimes = []
            for ii in range(nb_vtks):
                data = gathered_data_and_bounds[ii][0]
                bounds = gathered_data_and_bounds[ii][1]
                vtktimes.append(data["TIME"])

                # redistribute data
                for qty in (
                    self.partQuantities + self.spaceTimeHeatmaps + self.oneC_oneVs
                ):
                    key = qty.key
                    qty.values.append(data[key])

                    if getattr(qty, "xqty", None) is not None:
                        qty.points.append(data[qty.xqty.key])

                # redistribute bounds
                computed_bounds = {}  # for LOG only
                if not self.userArgs.noBounds and ii > 5:
                    for qty in self.all_movies:
                        if qty.key in bounds:
                            bound_low, bound_up = bounds[qty.key]
                            if qty.bounds[0] is None or bound_low < qty.bounds[0]:
                                qty.bounds[0] = bound_low
                            if qty.bounds[1] is None or bound_up > qty.bounds[1]:
                                qty.bounds[1] = bound_up
                            computed_bounds[qty.key] = qty.bounds

            if len(computed_bounds) > 0:
                LOG("Bounds computed:")
                for key in computed_bounds:
                    LOG(
                        f"{key:>10}: {computed_bounds[key][0]:.1e} {computed_bounds[key][1]:.1e}"
                    )
            LOG("Final Bounds:")
            for qtyInfo in self.all_movies:
                b1 = None if qtyInfo.bounds[0] is None else f"{qtyInfo.bounds[0]:.1e}"
                b2 = None if qtyInfo.bounds[1] is None else f"{qtyInfo.bounds[1]:.1e}"
                LOG(f"{qtyInfo.key:>10} {b1} {b2}")

            self.processor.set_vtktimes(vtktimes)

            if self.particles_requested and len(self.context.active_directions) >= 2:
                # cartesian for pcolormesh
                self.processor.partsInfo.parts_Z.set_data(
                    *convertGrid_toXZ(
                        self.processor.partsInfo.parts_X1.values,
                        self.processor.partsInfo.parts_X2.values,
                        self.context.geometry,
                    )
                )

        # delegate the render of all this stuff to the Renderer
        self.renderer.set_infos(self.processor.gridInfo, self.processor.partsInfo)
        self.renderer.render()

    def _name_frames(self):
        context = self.context

        self.slice1_list = context.get_slice1_vtkFiles()
        self.vtkList = context.get_global_vtkFiles()
        self.partList = context.get_particles_vtkFiles()

    def _apply_config(self):
        if self.userArgs.onlyMovie or self.userArgs.onlyAnalysis:
            return

        # gathering bounds for movies
        all_movies = [*self.linemovies1D, *self.mapmovies2D]
        self.all_movies = all_movies
        config = self.context.config

        LOG(f"config.json file requested: {config}")
        remaining_fields_tobound = set()

        if not self.userArgs.noBounds:
            for movie in all_movies:
                if not movie.bounds_set:
                    if movie.key not in config or "bounds" not in config[movie.key]:
                        remaining_fields_tobound.add(movie.key)
            if len(remaining_fields_tobound) > 0:
                LOG("Fields to bound: ", remaining_fields_tobound)
            else:
                LOG("All fields are already bounded in config")
        else:
            LOG("Bounds computation discarded.")

        for qtyInfo in self.quantities:
            AVAILABLE_KWARGS = [
                "bounds",
                "symbol",
                "title",
                "style_kwargs",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "xscale",
                "yscale",
                "norm",
            ]
            if qtyInfo.key in config:
                for key in config[qtyInfo.key]:
                    if key in AVAILABLE_KWARGS:
                        if key == "norm":
                            qtyInfo.set_norm(config[qtyInfo.key][key])
                        else:
                            setattr(qtyInfo, key, config[qtyInfo.key][key])

        return remaining_fields_tobound
