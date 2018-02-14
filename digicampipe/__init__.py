from pkg_resources import resource_string

__version__ = resource_string('digicampipe', 'VERSION').decode().strip()
