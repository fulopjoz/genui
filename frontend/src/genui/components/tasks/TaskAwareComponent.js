import React from "react";
import withUnmounted from '@ishawnwang/withunmounted'

class TaskAwareComponent extends React.Component {
  abort = new AbortController();
  hasUnmounted = false;


  constructor(props) {
    super(props);

    this.intervalID = null;
    this.actions = [];
    this.state = {
      tasks : {
        completed : [],
        running : [],
        errors : []
      }
      , tasksRunning : false
      , tasksUpToDate : false
    }
  }

  groupTasks = (data) => {
    const tasks = data;
    const completed = [];
    const running = [];
    const errors = [];
    Object.keys(tasks).forEach(task_name => {
      tasks[task_name].forEach(task => {
        task.task_name = task_name;
        if (task.status === 'SUCCESS') {
          completed.push(task)
        } else if (['STARTED', 'RECEIVED', 'PENDING', 'RETRY', 'PROGRESS'].includes(task.status)) {
          running.push(task)
        } else if (['FAILURE', 'REVOKED'].includes(task.status)) {
          errors.push(task)
        }
      });
    });

    return {
      completed : completed,
      running : running,
      errors : errors
    }
  };

  componentWillUnmount() {
    clearTimeout(this.intervalID);
  }

  componentDidMount() {
    this.fetchTasks();
  }

  fetchTasks = () => {
    fetch(this.props.tasksURL, {signal : this.abort.signal})
      .then(response => this.props.handleResponseErrors(response, 'Failed to fetch task info from backend.'))
      .then(data => {
        const tasks = this.groupTasks(data);
        this.updateTasks(tasks);
        this.intervalID = setTimeout(this.fetchTasks, 5000);
      }).catch(
      (error) => console.log(error)
    )
  };

  updateTasks = (groupedTasks) => {
    if (this.hasUnmounted) {
      return
    }

    // TODO: check if tasks changed when compared to previous state -> cancel state update if they are the same
    if (groupedTasks.running.length > 0) {
      this.setState({
        tasksRunning : true,
        tasksUpToDate : false,
        tasks : groupedTasks
      });
    } else {
      this.setState({
        tasksRunning : false,
        tasksUpToDate : true,
        tasks : groupedTasks
      });
    }

    while (this.actions.length !== 0) {
      this.actions.pop()(groupedTasks);
    }
  };

  registerTaskUpdateAction = (action) => {
    this.actions.push(action);
  };

  render() {
    return <React.Fragment>{this.props.render(this.state, this.registerTaskUpdateAction)}</React.Fragment>;
  }
}

export default withUnmounted(TaskAwareComponent);