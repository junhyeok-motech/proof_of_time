from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import solver
from inspect_ai.tool import bash_session


@solver
def uname_solver():
    async def solve(state, generate):
        from inspect_ai.util._sandbox.context import sandbox

        env = sandbox()
        result = await env.exec(["uname", "-s"])
        state.output.completion = result.stdout.strip()
        return state

    return solve


@task()
def docker_uname_task() -> Task:
    dataset = MemoryDataset([Sample(input="uname?", target="Linux")])
    return Task(dataset=dataset, solver=uname_solver(), sandbox="docker")


@solver
def bash_session_solver():
    async def solve(state, generate):
        session = bash_session()
        output = await session(action="type_submit", input="uname -s")
        state.output.completion = output.strip()
        return state

    return solve


@task()
def docker_bash_session_task() -> Task:
    dataset = MemoryDataset([Sample(input="uname via bash session?", target="Linux")])
    return Task(dataset=dataset, solver=bash_session_solver(), sandbox="docker")

