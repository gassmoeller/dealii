//----------------------------  mesh_3d_1.cc  ---------------------------
//    $Id$
//    Version: $Name$
//
//    Copyright (C) 2003 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  mesh_3d_1.cc  ---------------------------


// the real reason why coarsening_3d failed: take three cells to form an
// L, and we end up with an external face between two of them, which
// however has edges that are shared by the two cells. this creates a
// major upheaval in the data structures later on (as testified by
// coarsening_3d), but we can check this fact much earlier already (done
// here)


#include <base/logstream.h>
#include <grid/tria.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <grid/grid_reordering.h>
#include <grid/grid_generator.h>

#include <fstream>



void create_coarse_grid (Triangulation<3> &coarse_grid)
{
  std::vector<Point<3> >    vertices;
  std::vector<CellData<3> > cells;
  SubCellData               sub_cell_data;
  
  const Point<3> outer_points[8] = { Point<3>(-1,0,0),
                                     Point<3>(-1,-1,0),
                                     Point<3>(0,-1,0),
                                     Point<3>(+1,-1,0),
                                     Point<3>(+1,0,0),
                                     Point<3>(+1,+1,0),
                                     Point<3>(0,+1,0),
                                     Point<3>(-1,+1,0) };

                                   // first the point in the middle
                                   // and the rest of those on the
                                   // upper surface
  vertices.push_back (Point<3>(0,0,0));
  for (unsigned int i=0; i<7; ++i)
    vertices.push_back (outer_points[i]);

                                   // then points on lower surface
  vertices.push_back (Point<3>(0,0,-1));
  for (unsigned int i=0; i<7; ++i)
    vertices.push_back (outer_points[i]
                        +
                        Point<3>(0,0,-1));

  const unsigned int n_vertices_per_surface = 8;
  Assert (vertices.size() == n_vertices_per_surface*2,
          ExcInternalError());
    
  const unsigned int connectivity[3][4]
    = { { 1, 2, 3, 0 },
        { 3, 4, 5, 0 },
        { 0, 5, 6, 7 } };
  for (unsigned int i=0; i<3; ++i)
    {
      CellData<3> cell;
      for (unsigned int j=0; j<4; ++j)
        {
          cell.vertices[j]   = connectivity[i][j];
          cell.vertices[j+4] = connectivity[i][j]+n_vertices_per_surface;
        }
      cells.push_back (cell);
    }

                                   // finally generate a triangulation
                                   // out of this
  GridReordering<3>::reorder_cells (cells);
  coarse_grid.create_triangulation (vertices, cells, sub_cell_data);
}


int main () 
{
  std::ofstream logfile("mesh_3d_1.output");
  deallog.attach(logfile);
  deallog.depth_console(0);

  Triangulation<3> coarse_grid;
  create_coarse_grid (coarse_grid);

                                   // output all lines and faces
  for (Triangulation<3>::active_cell_iterator cell=coarse_grid.begin_active();
       cell != coarse_grid.end(); ++cell)
    {
      deallog << "Cell = " << cell << std::endl;
      for (unsigned int i=0; i<GeometryInfo<3>::lines_per_cell; ++i)
        deallog << "    Line = " << cell->line(i)
                << " : " << cell->line(i)->vertex_index(0)
                << " -> " << cell->line(i)->vertex_index(1)
                << std::endl;

      for (unsigned int i=0; i<GeometryInfo<3>::quads_per_cell; ++i)
        deallog << "    Quad = " << cell->quad(i)
                << " : " << cell->quad(i)->vertex_index(0)
                << " -> " << cell->quad(i)->vertex_index(1)
                << " -> " << cell->quad(i)->vertex_index(2)
                << " -> " << cell->quad(i)->vertex_index(3)
                << std::endl;
    }
}

  
  
