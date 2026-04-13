# nobody was executed in the making of this file
START_CARD_IDs = []
class WorkflowExecution:
    """Class to handle work flow execution, in a CLI friendly way
    """
    def __init__(self):
        self.workflow = None
        self.data_cache = dict()
    def set_workflow(self, cards, connections):
        # create the workflow tree from the cards and connections
        pass