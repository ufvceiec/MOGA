import logging
import numpy as np
import pandas as pd
from jmetal.lab.visualization.chord_plot import chord_diagram
from jmetal.lab.visualization.plotting import Plot
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from plotly import io as pio
from plotly import offline
from typing import TypeVar, List

S = TypeVar('S')
class MostrarFrente():
    
    def __init__(self, frente: List[S] = None,
                         axis_labels: list = None):
        self.frente = frente
        self.labels = axis_labels

    def mostrar_frente_2d(self, labels: List[str], filename: str = None, format: str = 'eps'):

            if len(labels) == 0 or len(labels) > 2:
                print("Introduzca 2 soluciones a mostrar")
                return
            indices = []
            for j in range(2):
                for i in range(len(self.labels)):
                    if labels[j] in self.labels[i] or labels[j] in self.objetivos[i]:
                        indices.append(i)


            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            for i in range(len(self.frente)):
                ax.scatter(self.frente[i].objectives[indices[0]], self.frente[i].objectives[indices[1]])
            ax.relim()
            ax.autoscale_view(True, True)

            ax.locator_params(nbins=4)

            if filename:
                plt.savefig(filename + '.' + format, format=format, dpi=1000)

            plt.show()
            plt.close(fig)

    def mostrar_frente_3d(self, labels: List[str], filename: str = None, format: str = 'eps'):

        if len(labels) != 3:
            print("Introduzca 3 soluciones a mostrar")
            return
        indices = []
        for j in range(3):
            for i in range(len(self.labels)):
                if labels[j] in self.labels[i] or labels[j] in self.labels[i]:
                    indices.append(i)

        if len(indices) != 3:
            print("labels incorrectos.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        for i in range(0, len(self.frente)):
            ax.scatter(self.frente[i].objectives[indices[0]],
                       self.frente[i].objectives[indices[1]],
                       self.frente[i].objectives[indices[2]])
        ax.relim()
        ax.autoscale_view(True, True)
        ax.view_init(elev=30.0, azim=15.0)
        ax.locator_params(nbins=4)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)

        plt.show()
        plt.close(fig)
        
class InteractivePlot(Plot):
    def __init__(self,
             plot_title: str = 'Pareto front approximation',
             reference_front: List[S] = None,
             reference_point: list = None,
             axis_labels: list = None):
        super(InteractivePlot, self).__init__(plot_title, reference_front, reference_point, axis_labels)
        self.figure = None
        self.layout = None
        self.data = []

    def plot(self, front, label=None, normalize: bool = False, labels: list= None, filename: str = None, format: str = 'HTML'):
        """ Plot a front of solutions (2D, 3D or parallel coordinates).

        :param front: List of solutions.
        :param label: Front name.
        :param normalize: Normalize the input front between 0 and 1 (for problems with more than 3 objectives).
        :param filename: Output filename.
        """
        if not isinstance(label, list):
            label = [label]
        
        self.layout = go.Layout(
            margin=dict(l=80, r=80, b=80, t=150),
            height=800,
            title='{}<br>{}'.format(self.plot_title, label[0]),
            scene=dict(
                xaxis=dict(title=labels[0] if labels[0] else None),
                yaxis=dict(title=labels[1] if labels[1] else None),
                
            ),
            hovermode='closest'
        )

        indices = []
        for j in range(2):
            for i in range(len(self.axis_labels)):
                if labels[j] in self.axis_labels[i] or labels[j] in self.axis_labels[i]:
                    indices.append(i)
        indices = np.array(indices)
        
        print("llega")
        # If any reference front, plot
        if self.reference_front:
            points, _ = self.get_points(self.reference_front)
            points = np.array(points)[:, indices]
            trace = self.__generate_trace(points=points, legend='Reference front', normalize=normalize,
                                          color='black', size=2)
            self.data.append(trace)

        # If any reference point, plot
        if self.reference_point:
            points = pd.DataFrame(self.reference_point)
            trace = self.__generate_trace(points=points, legend='Reference point', color='red', size=8)
            self.data.append(trace)

        # Get points and metadata
        points, _ = self.get_points(front)
        
        points = np.array(points)[:, indices]
        metadata = list(solution.__str__() for solution in front)

        trace = self.__generate_trace(points=points, metadata=metadata, legend='Front approximation',
                                      normalize=normalize)
        self.data.append(trace)
        self.figure = go.Figure(data=self.data, layout=self.layout)
        
        # Plot the figure
        if filename:
            if format == 'HTML':
                self.export_to_html(filename)
            else:
                pio.write_image(self.figure, filename + '.' + format)

    def export_to_html(self, filename: str) -> str:
        """ Export the graph to an interactive HTML (solutions can be selected to show some metadata).

        :param filename: Output file name.
        :return: Script as string. """
        html_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8"/>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://unpkg.com/sweetalert2@7.7.0/dist/sweetalert2.all.js"></script>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            </head>
            <body>
                ''' + self.export_to_div(filename=None, include_plotlyjs=False) + '''
                <script>                
                    var myPlot = document.querySelectorAll('div')[0];
                    myPlot.on('plotly_click', function(data){
                        var pts = '';

                        for(var i=0; i < data.points.length; i++){
                            pts = '(x, y) = ('+data.points[i].x +', '+ data.points[i].y.toPrecision(4)+')';
                            cs = data.points[i].customdata
                        }

                        if(typeof cs !== "undefined"){
                            swal({
                              title: 'Closest solution clicked:',
                              text: cs,
                              type: 'info',
                              position: 'bottom-end'
                            })
                        }
                    });

                    window.onresize = function() {
                       Plotly.Plots.resize(myPlot);
                    };
                </script>
            </body>
        </html>'''

        with open(filename + '.html', 'w') as outf:
            outf.write(html_string)

        return html_string

    def export_to_div(self, filename=None, include_plotlyjs: bool = False) -> str:
        """ Export as a `div` for embedding the graph in an HTML file.

        :param filename: Output file name (if desired, default to None).
        :param include_plotlyjs: If True, include plot.ly JS script (default to False).
        :return: Script as string.
        """
        script = offline.plot(self.figure, output_type='div', include_plotlyjs=include_plotlyjs, show_link=False)

        if filename:
            with open(filename + '.html', 'w') as outf:
                outf.write(script)

        return script

    def __generate_trace(self, points: pd.DataFrame, legend: str, metadata: list = None, normalize: bool = False,
                         **kwargs):
        dimension = points.shape[1]
        print(dimension)
        # tweak points size for 3D plots
        marker_size = 8
        if dimension == 3:
            marker_size = 4

        # if indicated, perform normalization
        if normalize:
            points = (points - points.min()) / (points.max() - points.min())

        marker = dict(
            color='#236FA4',
            size=marker_size,
            symbol='circle',
            line=dict(
                color='#236FA4',
                width=1
            ),
            opacity=0.8
        )
        marker.update(**kwargs)

        if dimension == 2:
            print("llega dimension 2")
            print(points[:, 0])
            trace = go.Scattergl(
                x=points[:,0],
                y=points[:,1],
                mode='markers',
                marker=marker,
                name=legend,
                customdata=metadata
            )
        elif dimension == 3:
            trace = go.Scatter3d(
                x=points[0],
                y=points[1],
                z=points[2],
                mode='markers',
                marker=marker,
                name=legend,
                customdata=metadata
            )
        else:
            dimensions = list()
            for column in points:
                dimensions.append(
                    dict(range=[0, 1],
                         label=self.axis_labels[column:column + 1][0] if self.axis_labels[column:column + 1] else None,
                         values=points[column])
                )

            trace = go.Parcoords(
                line=dict(
                    color='#236FA4'
                ),
                dimensions=dimensions,
                name=legend,
            )

        return trace

    
    
    

    