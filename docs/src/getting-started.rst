.. currentmodule:: singlestoredb

.. ipython:: python
   :suppress:

   import singlestoredb as s2
   conn = s2.connect()

Getting Started
===============

Database connections can be made using either keyword parameters or
a URL as described in the following sections.


Connect using DB-API Parameters
-------------------------------

Connections to SingleStoreDB can be made using the parameters described in
the `Python DB-API <https://peps.python.org/pep-0249/>`_. The ``host=``
parameter can be either a hostname or IP address (it can also be a
URL as shown in the following section). The ``port=`` parameter is an
integer value of the database server port number. The ``user=`` and
``password=`` parameters specify the database user credentials. The
``database=`` parameter, optionally, specifies the name of the database
to connect to.

A full list of connection parameters can be seen in the API documentation
for the :func:`singlestoredb.connect` function.

.. ipython:: python
   :verbatim:

   import singlestoredb as s2
   conn = s2.connect(host='...', port='...', user='...',
                     password='...', database='...')

Connect using a URL
-------------------

In addition, you can user a URL like in the SQLAlchemy package.

.. ipython:: python
   :verbatim:

   conn = s2.connect('user:password@host:port/database')

URLs work equally well to connect to the
`Data API <https://docs.singlestore.com/managed-service/en/reference/data-api.html>`_.

.. ipython:: python
   :verbatim:

   conn = s2.connect('https://user:password@host:port/database')

Specifying Additional Connection Parameters
-------------------------------------------

Connection parameters can be set either in the URL or
as parameters. Here ``local_infile=`` is set as a URL parameter.

.. ipython:: python
   :verbatim:

   conn = s2.connect('https://user:password@host:port/database?local_infile=True')


In this example, ``local_infile=`` and user credentials are specified
as keyword parameters.

.. ipython:: python
   :verbatim:

   conn = s2.connect('https://host:port/database', user='...', password='...',
                     local_infile=True)



Executing Queries
-----------------

Once you have a connection established, you can query the database.
As defined in the DB-API, a cursor is used to execute queries and fetch
the results.

.. ipython:: python

   with conn.cursor() as cur:
        cur.execute('show variables like "auto%"')
        for row in cur.fetchall():
            print(row)


Parameter Substitution
......................

If your queries require parameter substitutions, they can be specified in
one of two formats: named (``%(name)s``) or positional (``%s``).

.. warning:: As of v0.5.0, the substition parameter has been changed from
   ``:1``, ``:2``, etc. for list parameters and ``:foo``, ``:bar``, etc.
   for dictionary parameters to ``%s`` and ``%(foo)s``, ``%(bar)s``, etc.
   respectively, to ease the transition from other MySQL Python packages.

Named Substitution
^^^^^^^^^^^^^^^^^^

When named parameters
are used, the data structure passed to the :meth:`Cursor.execute` method
must be a dictionary, where the keys map to the names given in the substitutions
and the values are the values to substitute.

In the example below, ``%(pattern)s`` is replaced with the value ``"auto%"``. All
escaping and quoting of the substituted data values is done automatically.

.. ipython:: python

   with conn.cursor() as cur:
       cur.execute('show variables like %(pattern)s', dict(pattern='auto%'))
       for row in cur.fetchall():
           print(row)


Positional Substitution
^^^^^^^^^^^^^^^^^^^^^^^

If positional parameters are used, the data structure passed to the
:meth:`Cursor.execute` method must be a list or tuple with the same
number of elements as there are ``%s`` values in the query string.

In the example below, ``%s`` is replaced with the value ``"auto%"``. All
escaping and quoting of the substituted data values is done automatically.

.. ipython:: python

   with conn.cursor() as cur:
       cur.execute('show variables like %s', ['auto%'])
       for row in cur.fetchall():
           print(row)


Fetching Results
----------------

Fetching results can be done in a number of ways. The DB-API specifies three methods
that can be used to fetch results: :meth:`Cursor.fetchone`, :meth:`Cursor.fetchall`,
and :meth:`Cursor.fetchmany`.

The :meth:`Cursor.fetchone` method fetches a single row of data returned by a query.
The :meth:`Cursor.fetchall` method fetches all of the results of a query. The
:meth:`Cursor.fetchmany` method fetches a specified number of rows to retrieve.
The choice of which one to use depends mostly on the expected size of the result.
If the result is expected to be fairly small, fetching the entire result in one
call may be fine. However, if the query result will be large enough to put a strain
on the client computer's memory, it may be a better idea to fetch smaller batches
using :meth:`Cursor.fetchmany`.

In additon to the DB-API methods for fetching results, a :class:`Cursor` can be
iterated over itself.

.. ipython:: python

   with conn.cursor() as cur:
       cur.execute('show variables like "auto%"')
       for row in cur:
           print(row)


Result Type
...........

In addition to being able to specify the amount of data to be retrieved, you can also
specify the data structure that the results are returned in. By default, each row of
data is a tuple with one element per column from the query. However, it is also possible
to get results back as named tuples or dicts.

Tuples (Default)
^^^^^^^^^^^^^^^^

.. ipython:: python

   with s2.connect(results_type='tuples') as conn:
       with conn.cursor() as cur:
           cur.execute('show variables like "auto%"')
           for row in cur.fetchall():
               print(row)


Named Tuples
^^^^^^^^^^^^

.. ipython:: python

   with s2.connect(results_type='namedtuples') as conn:
       with conn.cursor() as cur:
           cur.execute('show variables like "auto%"')
           for row in cur.fetchall():
               print(row)


Dictionaries
^^^^^^^^^^^^

.. ipython:: python

   with s2.connect(results_type='dicts') as conn:
       with conn.cursor() as cur:
           cur.execute('show variables like "auto%"')
           for row in cur.fetchall():
               print(row)
