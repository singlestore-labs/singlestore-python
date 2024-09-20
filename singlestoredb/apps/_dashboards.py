import typing
import urllib.parse

from ._config import AppConfig
from ._process import kill_process_by_port

if typing.TYPE_CHECKING:
    from plotly.graph_objs import Figure


def run_dashboard_app(
    figure: 'Figure',
    debug: bool = False,
    kill_existing_app_server: bool = True,
) -> None:
    try:
        import dash
    except ImportError:
        raise ImportError('package dash is required to run dashboards')

    try:
        from plotly.graph_objs import Figure
    except ImportError:
        raise ImportError('package dash is required to run dashboards')

    if not isinstance(figure, Figure):
        raise TypeError('figure is not an instance of plotly Figure')

    app_config = AppConfig.from_env()

    if kill_existing_app_server:
        kill_process_by_port(app_config.listen_port)

    base_path = urllib.parse.urlparse(app_config.url).path

    app = dash.Dash(requests_pathname_prefix=base_path)
    app.layout = dash.html.Div(
        [
            dash.dcc.Graph(figure=figure),
        ],
    )

    app.run(
        host='0.0.0.0',
        debug=debug,
        port=str(app_config.listen_port),
        jupyter_mode='external',
    )

    if app_config.running_interactively:
        print(f'Dash app available at {app_config.url}')
