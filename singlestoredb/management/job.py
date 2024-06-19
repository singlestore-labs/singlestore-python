#!/usr/bin/env python
"""SingleStoreDB Cloud Scheduled Notebook Job."""
import datetime
import time
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from ..exceptions import ManagementError
from .manager import Manager
from .utils import camel_to_snake
from .utils import from_datetime
from .utils import get_cluster_id
from .utils import get_database_name
from .utils import get_workspace_id
from .utils import is_virtual_workspace
from .utils import to_datetime
from .utils import to_datetime_strict
from .utils import vars_to_str


class TargetType(Enum):
    WORKSPACE = 'Workspace'
    CLUSTER = 'Cluster'
    VIRTUAL_WORKSPACE = 'VirtualWorkspace'

    @classmethod
    def from_str(cls, s: str) -> 'TargetType':
        try:
            return cls[str(camel_to_snake(s)).upper()]
        except KeyError:
            raise ValueError(f'Unknown TargetType: {s}')

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)


class Status(Enum):
    UNKNOWN = 'Unknown'
    SCHEDULED = 'Scheduled'
    RUNNING = 'Running'
    COMPLETED = 'Completed'
    FAILED = 'Failed'
    ERROR = 'Error'
    CANCELED = 'Canceled'

    @classmethod
    def from_str(cls, s: str) -> 'Status':
        try:
            return cls[str(camel_to_snake(s)).upper()]
        except KeyError:
            return cls.UNKNOWN

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)


class ExecutionMetadata(object):

    avg_duration_in_seconds: Optional[float]
    count: int
    max_duration_in_seconds: Optional[float]
    status: Status

    def __init__(
        self,
        avg_duration_in_seconds: Optional[float],
        count: int,
        max_duration_in_seconds: Optional[float],
        status: Status,
    ):
        self.avg_duration_in_seconds = avg_duration_in_seconds
        self.count = count
        self.max_duration_in_seconds = max_duration_in_seconds
        self.status = status

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'ExecutionMetadata':
        """
        Construct an ExecutionMetadata from a dictionary of values.

        Parameters
        ----------
        obj : dict
            Dictionary of values

        Returns
        -------
        :class:`ExecutionMetadata`

        """
        out = cls(
            avg_duration_in_seconds=obj.get('avgDurationInSeconds'),
            count=obj['count'],
            max_duration_in_seconds=obj.get('maxDurationInSeconds'),
            status=Status.from_str(obj['status']),
        )

        return out

    def __str__(self) -> str:
        """Return string representation."""
        return vars_to_str(self)

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)


class ExecutionConfig(object):

    create_snapshot: bool
    max_duration_in_mins: int
    notebook_path: str

    def __init__(
        self,
        create_snapshot: bool,
        max_duration_in_mins: int,
        notebook_path: str,
    ):
        self.create_snapshot = create_snapshot
        self.max_duration_in_mins = max_duration_in_mins
        self.notebook_path = notebook_path

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'ExecutionConfig':
        """
        Construct an ExecutionConfig from a dictionary of values.

        Parameters
        ----------
        obj : dict
            Dictionary of values

        Returns
        -------
        :class:`ExecutionConfig`

        """
        out = cls(
            create_snapshot=obj['createSnapshot'],
            max_duration_in_mins=obj['maxAllowedExecutionDurationInMinutes'],
            notebook_path=obj['notebookPath'],
        )

        return out

    def __str__(self) -> str:
        """Return string representation."""
        return vars_to_str(self)

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)


class Schedule(object):

    execution_interval_in_minutes: Optional[int]
    mode: str
    start_at: Optional[datetime.datetime]

    def __init__(
        self,
        execution_interval_in_minutes: Optional[int],
        mode: str,
        start_at: Optional[datetime.datetime],
    ):
        self.execution_interval_in_minutes = execution_interval_in_minutes
        self.mode = mode
        self.start_at = start_at

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'Schedule':
        """
        Construct a Schedule from a dictionary of values.

        Parameters
        ----------
        obj : dict
            Dictionary of values

        Returns
        -------
        :class:`Schedule`

        """
        out = cls(
            execution_interval_in_minutes=obj.get('executionIntervalInMinutes'),
            mode=obj['mode'],
            start_at=to_datetime(obj.get('startAt')),
        )

        return out

    def __str__(self) -> str:
        """Return string representation."""
        return vars_to_str(self)

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)


class TargetConfig(object):

    database_name: Optional[str]
    resume_target: bool
    target_id: str
    target_type: TargetType

    def __init__(
        self,
        database_name: Optional[str],
        resume_target: bool,
        target_id: str,
        target_type: TargetType,
    ):
        self.database_name = database_name
        self.resume_target = resume_target
        self.target_id = target_id
        self.target_type = target_type

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'TargetConfig':
        """
        Construct a TargetConfig from a dictionary of values.

        Parameters
        ----------
        obj : dict
            Dictionary of values

        Returns
        -------
        :class:`TargetConfig`

        """
        out = cls(
            database_name=obj.get('databaseName'),
            resume_target=obj['resumeTarget'],
            target_id=obj['targetID'],
            target_type=TargetType.from_str(obj['targetType']),
        )

        return out

    def __str__(self) -> str:
        """Return string representation."""
        return vars_to_str(self)

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)


class Job(object): # TODO: Check which fields are optional
    """
    Scheduled Notebook Job definition.

    This object is not directly instantiated. It is used in results
    of API calls on the :class:`JobsManager`. See :meth:`JobsManager.run`.
    """

    completed_executions_count: int
    created_at: datetime.datetime
    description: Optional[str]
    enqueued_by: str
    execution_config: ExecutionConfig
    job_id: str
    job_metadata: List[ExecutionMetadata]
    name: Optional[str]
    schedule: Schedule
    target_config: Optional[TargetConfig]
    terminated_at: Optional[datetime.datetime]

    def __init__(
        self,
        completed_executions_count: int,
        created_at: datetime.datetime,
        description: Optional[str],
        enqueued_by: str,
        execution_config: ExecutionConfig,
        job_id: str,
        job_metadata: List[ExecutionMetadata],
        name: Optional[str],
        schedule: Schedule,
        target_config: Optional[TargetConfig],
        terminated_at: Optional[datetime.datetime],
    ):
        self.completed_executions_count = completed_executions_count
        self.created_at = created_at
        self.description = description
        self.enqueued_by = enqueued_by
        self.execution_config = execution_config
        self.job_id = job_id
        self.job_metadata = job_metadata
        self.name = name
        self.schedule = schedule
        self.target_config = target_config
        self.terminated_at = terminated_at
        self._manager: Optional[JobsManager] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any], manager: 'JobsManager') -> 'Job':
        """
        Construct a Job from a dictionary of values.

        Parameters
        ----------
        obj : dict
            Dictionary of values

        Returns
        -------
        :class:`Job`

        """

        out = cls(
            completed_executions_count=obj['completedExecutionsCount'],
            created_at=to_datetime_strict(obj['createdAt']),
            description=obj.get('description'),
            enqueued_by=obj['enqueuedBy'],
            execution_config=ExecutionConfig.from_dict(obj['executionConfig']),
            job_id=obj['jobID'],
            job_metadata=[ExecutionMetadata.from_dict(x) for x in obj['jobMetadata']],
            name=obj.get('name'),
            schedule=Schedule.from_dict(obj['schedule']),
            target_config=TargetConfig.from_dict(obj['targetConfig']) if 'targetConfig' in obj else None,
            terminated_at=to_datetime(obj.get('terminatedAt')),
        )
        out._manager = manager
        return out

    def wait(self, timeout: Optional[int] = None) -> None:
        """Wait for the job to complete."""
        if self._manager is None:
            raise ManagementError(msg='Job not initialized with JobsManager')
        self._manager.__wait_for_job__(self, timeout)

    def __str__(self) -> str:
        """Return string representation."""
        return vars_to_str(self)

    def __repr__(self) -> str:
        """Return string representation."""
        return str(self)


class JobsManager(object):
    """
    TODO: Add more details about the job manager
    """

    def __init__(self, manager: Optional[Manager]):
        self._manager = manager

    def run(
        self,
        notebook_path: str,
        pool_name: Optional[str] = None,
    ) -> Job:
        """Creates and returns the created scheduled notebook job."""
        if self._manager is None:
            raise ManagementError(msg='JobsManager not initialized')

        database_name = get_database_name()
        if database_name is not None and database_name != '':
            target_config = dict(
                resumeTarget=False,
                databaseName=database_name,
            ) # type: Dict[str, Any]

            workspace_id = get_workspace_id()
            cluster_id = get_cluster_id()
            if workspace_id is not None:
                target_config['targetID'] = workspace_id
                if is_virtual_workspace():
                    target_config['targetType'] = TargetType.VIRTUAL_WORKSPACE.value
                else:
                    target_config['targetType'] = TargetType.WORKSPACE.value

            elif cluster_id is not None:
                target_config['targetID'] = cluster_id
                if is_virtual_workspace():
                    target_config['targetType'] = TargetType.VIRTUAL_WORKSPACE.value
                else:
                    target_config['targetType'] = TargetType.CLUSTER.value


        job_run_json = dict(
            executionConfig=dict(
                createSnapshot=False,
                notebookPath=notebook_path,
            ),
            schedule=dict(
                mode='Once',
                startAt=from_datetime(datetime.datetime.now()),
            ),
            targetConfig=target_config,
        )  # type: Dict[str, Any]

        if pool_name is not None:
            job_run_json['poolName'] = pool_name

        print(job_run_json)

        res = self._manager._post('jobs', json=job_run_json).json()
        return Job.from_dict(res, self)

    def wait(self, jobs: List[Union[str, Job]], timeout: Optional[int] = None) -> None:
        if timeout is not None:
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

        for job in jobs:
            self.__wait_for_job__(job, int((finish_time - datetime.datetime.now()).total_seconds()) if timeout is not None else None)

    def __wait_for_job__(self, job: Union[str, Job], timeout: Optional[int] = None) -> None:
        if self._manager is None:
            raise ManagementError(msg='JobsManager not initialized')

        if timeout is not None:
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

        if isinstance(job, str):
            job_id = job
        else:
            job_id = job.job_id

        while True:
            if timeout is not None and datetime.datetime.now() > finish_time:
                raise TimeoutError(f'Timeout waiting for job {job_id}')

            res = self._manager._get(f'jobs/{job_id}').json()
            job = Job.from_dict(res, self)
            if job.schedule.mode == 'Once' and job.completed_executions_count > 0:
                return
            if job.schedule.mode == 'Recurring':
                raise ValueError(f'Cannot wait for recurring job {job_id}')
            time.sleep(1)

    def get(self, job_id: str) -> Job:
        """Get a job by its ID."""
        if self._manager is None:
            raise ManagementError(msg='JobsManager not initialized')

        res = self._manager._get(f'jobs/{job_id}').json()
        return Job.from_dict(res, self)

    def delete(self, job_id: str) -> bool:
        """Delete a job by its ID."""
        if self._manager is None:
            raise ManagementError(msg='JobsManager not initialized')

        return self._manager._delete(f'jobs/{job_id}').json()
